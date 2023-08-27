
from http import HTTPStatus
import logging
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

class InvalidInput(ValueError):
    """
    Exception class indicating invalid input arguments.
    HTTP Servers should return HTTP_400 (Bad Request).
    """

    def __init__(self, reason):
        self.reason = reason

    def __str__(self):
        return self.reason

async def invalid_input_handler(_, exc):
    logger.error("Exception:", exc_info=exc)
    return JSONResponse(status_code=HTTPStatus.BAD_REQUEST, content={"error": str(exc)})
