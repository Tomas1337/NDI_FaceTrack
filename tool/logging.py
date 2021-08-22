import logging
#Setup Logging

class DuplicateFilter(logging.Filter):
<<<<<<< HEAD

=======
>>>>>>> development
    def filter(self, record):
        # add other fields if you need more granular comparison, depends on your app
        current_log = (record.module, record.levelno, record.msg)
        if current_log != getattr(self, "last_log", None):
            self.last_log = current_log
            return True
        return False

def add_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    c_handler = logging.StreamHandler()
    c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    c_handler.setLevel(logging.INFO)
    c_handler.setFormatter(c_format)

    f_handler = logging.FileHandler('debug.log')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_handler.setLevel(logging.DEBUG)
    f_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
<<<<<<< HEAD
    logger.addFilter(DuplicateFilter()) 
=======
    #logger.addFilter(DuplicateFilter()) 
>>>>>>> development

    return logger
