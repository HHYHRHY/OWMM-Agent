"""
This file contains an example of a multi-processing server that listens to multiple clients and updates the environment based on the actions received from the clients.
"""

import multiprocessing
from multiprocessing import Queue
from multiprocessing.connection import Listener
import threading
import habitat_sim 
from habitat import Env
from multiprocessing.connection import Client
import time

class ActionServer:
    def __init__(self, env: Env, *args, **kwargs):
        # register habitat environment
        self.env = env
        
        self.action_queue = Queue()

    def update(self):
        while True:
            # set a fixed time interval to update the environment
            time.sleep(0.5)
            if not self.action_queue.empty(): 
                action = self.action_queue.get()
                print(action)
                # observation = self.env.update(action)
                
                # Assuming the producer is connected via a connection object
                # conn.send(observation)

if __name__ == "__main__":
    action_queue = multiprocessing.Queue()
    server = ActionServer(action_queue)

    def send_dummy_messages(client_name, action_queue):
        while True:
            message = f"{client_name}: Dummy message"
            action_queue.put(message)
            time.sleep(1)  # wait for 1 second before sending the next message

    # Start threads for each client
    multiprocessing.Process(target=send_dummy_messages, args=("client_1", server.action_queue)).start()
    multiprocessing.Process(target=send_dummy_messages, args=("client_2", server.action_queue)).start()

    while True:
        server.update()
