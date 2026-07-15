import hmac
import logging
import os
import base64
import traceback as tb
from typing import Dict, List, Optional

from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.routing import APIRoute

from .response import ModelResponse
from .model import LabelStudioMLBase
from .exceptions import AnswerException, answer

logger = logging.getLogger(__name__)


class LoggingRoute(APIRoute):
    def get_route_handler(self):
        original_handler = super().get_route_handler()
        async def custom_route_handler(request: Request) -> Response:
            logger.debug('Request headers: %s', request.headers)
            if request.method in ("POST", "PUT", "PATCH"):
                body = await request.body()
                logger.debug('Request body: %s', body)
            
            response = await original_handler(request)
            
            logger.debug('Response status: %s', response.status_code)
            logger.debug('Response headers: %s', response.headers)
            if hasattr(response, "body"):
                logger.debug('Response body: %s', response.body)
            return response
        return custom_route_handler


MODEL_CLASS = LabelStudioMLBase
BASIC_AUTH = None
app = None


def safe_str_cmp(a, b):
    return hmac.compare_digest(a, b)


def init_app(model_class, basic_auth_user=None, basic_auth_pass=None):
    global MODEL_CLASS
    global BASIC_AUTH
    global app

    if not issubclass(model_class, LabelStudioMLBase):
        raise ValueError('Inference class should be the subclass of ' + LabelStudioMLBase.__class__.__name__)

    MODEL_CLASS = model_class
    basic_auth_user = basic_auth_user or os.environ.get('BASIC_AUTH_USER')
    basic_auth_pass = basic_auth_pass or os.environ.get('BASIC_AUTH_PASS')
    if basic_auth_user and basic_auth_pass:
        BASIC_AUTH = (basic_auth_user, basic_auth_pass)

    app = FastAPI(title="Label Studio ML Backend")
    app.router.route_class = LoggingRoute

    # Backwards compatibility for app.run() and app.test_client()
    def fastapi_run(host='0.0.0.0', port=9090, debug=False, **kwargs):
        import uvicorn
        uvicorn.run(app, host=host, port=port)
    app.run = fastapi_run

    app.config = {}

    def test_client():
        from fastapi.testclient import TestClient
        
        class CompatibleTestClient(TestClient):
            def request(self, method, url, *args, **kwargs):
                content_type = kwargs.pop('content_type', None)
                if content_type:
                    headers = kwargs.setdefault('headers', {})
                    headers['Content-Type'] = content_type
                data = kwargs.get('data')
                if data is not None and isinstance(data, (str, bytes)):
                    kwargs['content'] = kwargs.pop('data')
                
                response = super().request(method, url, *args, **kwargs)
                response.data = response.content
                response.get_json = lambda: response.json()
                return response

            def post(self, url, *args, **kwargs):
                content_type = kwargs.pop('content_type', None)
                if content_type:
                    headers = kwargs.setdefault('headers', {})
                    headers['Content-Type'] = content_type
                data = kwargs.get('data')
                if data is not None and isinstance(data, (str, bytes)):
                    kwargs['content'] = kwargs.pop('data')
                
                response = super().post(url, *args, **kwargs)
                response.data = response.content
                response.get_json = lambda: response.json()
                return response

            def get(self, url, *args, **kwargs):
                content_type = kwargs.pop('content_type', None)
                if content_type:
                    headers = kwargs.setdefault('headers', {})
                    headers['Content-Type'] = content_type
                
                response = super().get(url, *args, **kwargs)
                response.data = response.content
                response.get_json = lambda: response.json()
                return response
                
        return CompatibleTestClient(app)
        
    app.test_client = test_client

    # Register exception handlers
    @app.exception_handler(AnswerException)
    def answer_exception_handler(request: Request, exc: AnswerException):
        traceback = tb.format_exc()
        logger.error(traceback)
        result = exc.result or {}
        if 'traceback' not in result:
            result['traceback'] = traceback
        return answer(exc.status, exc.msg, result, request)

    @app.exception_handler(FileNotFoundError)
    def file_not_found_handler(request: Request, exc: FileNotFoundError):
        logger.warning('Got error: ' + str(exc))
        return PlainTextResponse(str(exc), status_code=404)

    @app.exception_handler(AssertionError)
    def assertion_error_handler(request: Request, exc: AssertionError):
        logger.error(str(exc), exc_info=True)
        return PlainTextResponse(str(exc), status_code=500)

    @app.exception_handler(IndexError)
    def index_error_handler(request: Request, exc: IndexError):
        logger.error(str(exc), exc_info=True)
        return PlainTextResponse(str(exc), status_code=500)

    @app.exception_handler(Exception)
    def general_exception_handler(request: Request, exc: Exception):
        traceback = tb.format_exc()
        logger.error(traceback)
        print(traceback)
        body = {'traceback': traceback}
        return answer(500, f"{exc.__class__.__name__}: {str(exc)}", body, request)

    # Basic Auth Middleware
    @app.middleware("http")
    async def check_auth_middleware(request: Request, call_next):
        if BASIC_AUTH is not None:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Basic "):
                return Response('Unauthorized', status_code=401, headers={'WWW-Authenticate': 'Basic realm="Login required"'})
            try:
                auth_decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
                username, password = auth_decoded.split(":", 1)
            except Exception:
                return Response('Unauthorized', status_code=401, headers={'WWW-Authenticate': 'Basic realm="Login required"'})
            
            if not (safe_str_cmp(username, BASIC_AUTH[0]) and safe_str_cmp(password, BASIC_AUTH[1])):
                return Response('Unauthorized', status_code=401, headers={'WWW-Authenticate': 'Basic realm="Login required"'})
                
        response = await call_next(request)
        return response

    # Define routes
    @app.post('/predict')
    def _predict(data: dict):
        tasks = data.get('tasks')
        label_config = data.get('label_config')
        project = str(data.get('project')) if data.get('project') is not None else None
        project_id = project.split('.', 1)[0] if project else None
        params = data.get('params', {}) or {}
        context = params.pop('context', {}) or {}

        model = MODEL_CLASS(project_id=project_id,
                            label_config=label_config)

        response = model.predict(tasks, context=context, **params)

        # if there is no model version we will take the default
        if isinstance(response, ModelResponse):
            if not response.has_model_version():
                mv = model.model_version
                if mv:
                    response.set_version(str(mv))
            else:
                response.update_predictions_version()

            response = response.model_dump()

        res = response
        if res is None:
            res = []

        if isinstance(res, dict):
            res = response.get("predictions", response)

        return {'results': res}

    @app.post('/setup')
    def _setup(data: dict):
        project = data.get('project')
        project_id = str(project).split('.', 1)[0] if project is not None else None
        label_config = data.get('schema')
        extra_params = data.get('extra_params')
        model = MODEL_CLASS(project_id=project_id,
                            label_config=label_config)

        if extra_params:
            model.set_extra_params(extra_params)

        model_version = model.get('model_version')
        return {'model_version': model_version}

    TRAIN_EVENTS = (
        'ANNOTATION_CREATED',
        'ANNOTATION_UPDATED',
        'ANNOTATION_DELETED',
        'START_TRAINING'
    )

    @app.post('/webhook')
    def webhook(data: dict):
        event = data.pop('action', None)
        if event not in TRAIN_EVENTS:
            return {'status': 'Unknown event'}
        
        project_id = str(data['project']['id'])
        label_config = data['project']['label_config']
        model = MODEL_CLASS(project_id, label_config=label_config)
        result = model.fit(event, data)

        try:
            return JSONResponse(status_code=201, content={'result': result, 'status': 'ok'})
        except Exception as e:
            return JSONResponse(status_code=201, content={'error': str(e), 'status': 'error'})

    @app.get('/health')
    @app.get('/')
    def health():
        return {
            'status': 'UP',
            'model_class': MODEL_CLASS.__name__
        }

    @app.get('/metrics')
    def metrics():
        return {}

    return app


def run_app(model_class, host='0.0.0.0', port=9090, log_level='info', **kwargs):
    import uvicorn
    fastapi_app = init_app(model_class)
    uvicorn.run(fastapi_app, host=host, port=port, log_level=log_level, **kwargs)

