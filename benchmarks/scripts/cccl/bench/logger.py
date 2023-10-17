import logging

class Logger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            logger = logging.getLogger()
            logger.setLevel(logging.DEBUG)
            file_handler = logging.FileHandler('cccl_meta_bench.log')
            file_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s'))
            logger.addHandler(file_handler)
            cls._instance.logger = logger

        return cls._instance

    def info(self, message):
        self.logger.info(message)
