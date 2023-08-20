import os
import sys
import abc
import queue
import time
import logging
import threading
from http import HTTPStatus
from xmlrpc.server import SimpleXMLRPCServer, SimpleXMLRPCRequestHandler

# importlib machinery needs to be available for importing client modules
from importlib.machinery import SourceFileLoader

logger = logging.getLogger(__name__)

EXECUTION_QUEUE = queue.Queue()
RETURN_VALUE_NAME = 'RPC_SERVER_RETURN_VALUE'
ERROR_VALUE_NAME = 'RPC_SERVER_ERROR_VALUE'


def run_in_main_thread(callable_instance, *args):
    """
    Runs the provided callable instance in the main thread by added it to a que
    that is processed by a recurring event in an integration like a timer.

    :param call callable_instance: A callable.
    :return: The return value of any call from the client.
    """
    timeout = int(os.environ.get('RPC_TIME_OUT', 60))

    globals().pop(RETURN_VALUE_NAME, None)
    globals().pop(ERROR_VALUE_NAME, None)
    EXECUTION_QUEUE.put((callable_instance, args))

    for attempt in range(timeout * 10):
        if RETURN_VALUE_NAME in globals():
            return globals().get(RETURN_VALUE_NAME)
        elif ERROR_VALUE_NAME in globals():
            raise globals()[ERROR_VALUE_NAME]
        else:
            time.sleep(0.1)

    # if RETURN_VALUE_NAME not in globals():
    #     raise TimeoutError(
    #         f'The call "{callable_instance.__name__}" timed out because it hit the timeout limit'
    #         f' of {timeout} seconds.'
    #     )


def execute_queued_calls(*extra_args):
    """
    Runs calls in the execution que till they are gone. Designed to be passed to a
    recurring event in an integration like a timer.
    """
    while not EXECUTION_QUEUE.empty():
        if RETURN_VALUE_NAME not in globals():
            callable_instance, args = EXECUTION_QUEUE.get()
            try:
                globals()[RETURN_VALUE_NAME] = callable_instance(*args)
            except Exception as error:
                # store the error in the globals and re-raise it
                globals()[ERROR_VALUE_NAME] = error
                raise error


class AuthenticatedRequestHandler(SimpleXMLRPCRequestHandler):
    def is_authorized(self):
        """
        Checks if the Authorization header matches the key generated by the server.

        :returns: Whether the request is authorized.
        :rtype: bool
        """
        # do not allow requests sent cross site
        if self.headers.get('Sec-Fetch-Site') == 'cross-site':
            return False
        # do not allow requests from another origin
        if self.headers.get('Origin'):
            return False
        return True

    def report_401(self):
        """
        Reports an unauthorized error back to the client.
        """
        self.send_response(HTTPStatus.UNAUTHORIZED)
        response = b'Not authorized'
        self.send_header("Content-type", "text/plain")
        self.send_header("Content-length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def do_POST(self):
        """
        Overrides the post method to implement authentication.
        """
        if self.is_authorized():
            super(AuthenticatedRequestHandler, self).do_POST()
        else:
            self.report_401()


class BaseServer(SimpleXMLRPCServer):
    def __init__(self, *args, **kwargs):
        kwargs['requestHandler'] = AuthenticatedRequestHandler
        super(BaseServer, self).__init__(*args, **kwargs)

    def serve_until_killed(self):
        """
        Serves till killed by the client.
        """
        self.quit = False
        while not self.quit:
            self.handle_request()


class BaseRPCServer:
    def __init__(self, name, port, is_thread=False):
        """
        Initialize the base server.

        :param str name: The name of the server.
        :param int port: The number of the server port.
        :param bool is_thread: Whether the server is encapsulated in a thread.
        """
        self.server = BaseServer((os.environ.get('RPC_HOST', '127.0.0.1'), port), logRequests=False, allow_none=True)
        self.is_thread = is_thread
        self.server.register_function(self.add_new_callable)
        self.server.register_function(self.kill)
        self.server.register_function(self.is_running)
        self.server.register_function(self.set_env)
        self.server.register_introspection_functions()
        self.server.register_multicall_functions()
        logger.info(f'Started RPC server "{name}".')

    @staticmethod
    def is_running():
        """
        Responds if the server is running.
        """
        return True

    @staticmethod
    def set_env(name, value):
        """
        Sets an environment variable in the server's python environment.

        :param str name: The name of the variable.
        :param str value: The value.
        """
        os.environ[name] = str(value)

    def kill(self):
        """
        Kill the running server from the client. Only if running in blocking mode.
        """
        self.server.quit = True
        return True

    def add_new_callable(self, callable_name, code, client_system_path, remap_pairs=None):
        """
        Adds a new callable defined in the client to the server.

        :param str callable_name: The name of the function that will be added to the server.
        :param str code: The code of the callable that will be added to the server.
        :param list[str] client_system_path: The list of python system paths from the client.
        :param list(tuple) remap_pairs: A list of tuples with first value being the client python path root and the
        second being the new server path root. This can be useful if the client and server are on two different file
        systems and the root of the import paths need to be dynamically replaced.
        :return str: A response message back to the client.
        """
        for path in client_system_path:
            # if a list of remap pairs are provided, they will be remapped before being added to the system path
            for client_path_root, matching_server_path_root in remap_pairs or []:
                if path.startswith(client_path_root):
                    path = os.path.join(
                        matching_server_path_root, path.replace(client_path_root, '').replace(os.sep, '/').strip('/')
                    )

            if path not in sys.path:
                sys.path.append(path)

        # run the function code
        exec(code)
        callable_instance = locals().copy().get(callable_name)

        # grab it from the locals and register it with the server
        if callable_instance:
            if self.is_thread:
                self.server.register_function(self.thread_safe_call(callable_instance), callable_name)
            else:
                self.server.register_function(callable_instance, callable_name)
        return f'The function "{callable_name}" has been successfully registered with the server!'


class BaseRPCServerThread(threading.Thread, BaseRPCServer):
    def __init__(self, name, port):
        """
        Initialize the base rpc server.

        :param str name: The name of the server.
        :param int port: The number of the server port.
        """
        threading.Thread.__init__(self, name=name, daemon=True)
        BaseRPCServer.__init__(self, name, port, is_thread=True)

    def run(self):
        """
        Overrides the run method.
        """
        self.server.serve_forever()

    @abc.abstractmethod
    def thread_safe_call(self, callable_instance, *args):
        """
        Implements thread safe execution of a call.
        """
        return


class BaseRPCServerManager:
    @abc.abstractmethod
    def __init__(self):
        """
        Initialize the server manager.
        Note: when this class is subclassed `name`, `port`, `threaded_server_class` need to be defined.
        """
        self.server_thread = None
        self.server_blocking = None

    def start_server_thread(self):
        """
        Starts the server in a thread.
        """
        self.server_thread = self.threaded_server_class(self.name, self.port)
        self.server_thread.start()

    def start_server_blocking(self):
        """
        Starts the server in the main thread, which blocks all other processes. This can only
        be killed by the client.
        """
        self.server_blocking = BaseRPCServer(self.name, self.port)
        self.server_blocking.server.serve_until_killed()

    def start(self, threaded=True):
        """
        Starts the server.

        :param bool threaded: Whether to start the server in a thread. If not threaded
        it will block all other processes.
        """
        # start the server in a thread
        if threaded and not self.server_thread:
            self.start_server_thread()

        # start the blocking server
        elif not threaded and not self.server_blocking:
            self.start_server_blocking()

        else:
            logger.info(f'RPC server "{self.name}" is already running...')

    def shutdown(self):
        """
        Shuts down the server.
        """
        if self.server_thread:
            logger.info(f'RPC server "{self.name}" is shutting down...')

            # kill the server in the thread
            if self.server_thread:
                self.server_thread.server.shutdown()
                self.server_thread.join()

            logger.info(f'RPC server "{self.name}" has shutdown.')
