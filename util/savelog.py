import os
import sys
from datetime import datetime

host_name = os.uname()[1]

# global setting
# set like set_save_log(True)
__save_log = False


def set_save_log(flag):
    global __save_log
    if flag:
        print("logging started:" + logger.get_dt())
    __save_log = flag
    if __save_log:
        print("input: python3", " ".join([str(x) for x in sys.argv]))
        print('env: running on:', host_name)
        # print('env: ', 'remote' if remote else 'local' ,' mode')
        print('file: ', logger.get_filename())

        # # print git commit id for log investigation
        # try:
        #     cmd = "git rev-parse HEAD"
        #     label = subprocess.check_output(cmd.split()).strip().decode('utf-8')
        # except Exception as ex:
        #     label = str(ex)
        # print('commit:', label)


def get_save_log():
    # may not be used
    # just True or False
    global __save_log
    return __save_log


# auto logging
class Tee(object):
    def __init__(self):
        self.stdout = sys.stdout
        self.__str_dt = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.__filename = 'logs/' + self.__str_dt + '_' + host_name + '.txt'

    def write(self, obj):
        self.stdout.write(obj)
        if get_save_log() == False:
            return
        with open(self.__filename, 'a') as f:
            f.write(obj)

    def flush(self):
        return

    def get_filename(self):
        return self.__filename

    def get_filehead(self):
        # without ext
        return self.__filename[:-4]

    def get_dt(self):
        return self.__str_dt


def get_filehead():
    # usd this and do not access to savelog.logger()
    global logger
    return logger.get_filehead()


# backup = sys.stdout
logger = Tee()
sys.stdout = logger

loaded = True
