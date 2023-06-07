import logging
logger = logging.getLogger(__name__)
FORMAT = "[%(levelname)s][%(asctime)s] %(filename)s:%(funcName)s:%(lineno)s: %(message)s"
logging.basicConfig(format=FORMAT)
logger.setLevel(logging.DEBUG)
