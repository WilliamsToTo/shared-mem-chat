#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.mturk.core.worlds import MTurkOnboardWorld, MTurkTaskWorld
from parlai.core.worlds import validate
from joblib import Parallel, delayed
import json
from mephisto.core.logger_core import get_logger
import os
from mephisto.core.utils import get_root_dir
import re

logger = get_logger(name=__name__, verbose=True, level="info")

# TODO get shared memory via assignment id
class MTurkMultiAgentDialogOnboardWorld(MTurkOnboardWorld):
    def __init__(self, opt, mturk_agent):
        super().__init__(opt, mturk_agent)
        self.opt = opt
        self.welcome = False
        
        logger.info("read dialogues")
        # read shared memory file 
        data_path = os.path.join(get_root_dir(), "mephisto/tasks/shared-mem-chat/empatheticdialogues/empathetic-dialogues-train.json")
        fr = open(data_path,"r")
        self.shared_dialogues = json.load(fr)
        fr.close()
        self.shared_memory = self.get_one_shared_memory()
        logger.info(self.shared_memory)
        #store dialogues
        fw = open(data_path, "w", encoding="utf-8")
        json.dump(self.shared_dialogues, fw, ensure_ascii=False, indent=4)
        fw.close()
        logger.info("restore dialogues")
        
        
        
    def parley(self):
        user = self.mturk_agent
        user.agent_id = "Onboarding Agent"
        
        if not self.welcome:
            user.observe({
                    "id": "System",
                    "text":"Welcome to onboard! This is a tutorial to help you get familiar with the task."\
                    "Firstly, please have a look at the task description on the left panel "\
                    "that will be very helpful to understand the task.\n(If you finish the reading, input any key "\
                    "into the chatbox to continue)"
                })
            x = self.mturk_agent.act(timeout=self.opt["turn_timeout"])
            self.welcome = True
        
        user.observe({
                    "id": "System",
                    "text": "Well done! As you can see, there will be another worker to interact with you and "\
                            "you will play different roles. In this onboarding process, I will play the role as 'User' "\
                            "and you will be 'Wizard'. Let's get started!\n(input any key to continue)"
                })
        x = self.mturk_agent.act(timeout=self.opt["turn_timeout"])
        
        user.observe({"id": "The topic happened before", "text": self.shared_memory})
        
        user.observe({
                    "id": "***User***",
                    "text": "Hello!"  
                })
        
        user.observe({
                    "id": "System",
                    "text": "Following, you will start the conversation with user."
                }) 
        x = self.mturk_agent.act(timeout=self.opt["turn_timeout"])
        
        user.observe({
                "id": "System",
                "text": "Now, you enter the first step of the workflow: greeting. You will see a dropdown list, there is one"\
                        "avaible template you can choose, you need to enter in the 'input', in the real task you"\
                        "response will be sent to user"\
                        "(In the real task, you will have more than one template to choose.)",
                
                "task_data": {
                    "respond_with_form":[
                        {
                            "type":"choices",
                            "question":"Greet",
                            "choices":["______, how's are you?" ],
                        }, 
                        {"type":"text","question":"Input"}, 
                    ]
                  }
             })
        
        x = self.mturk_agent.act(timeout=self.opt["turn_timeout"])
        
        user.observe({
                    "id": "***User***",
                    "text": "I feel so sad. I can't forget her."  
                })
     
        user.observe({
                "id": "System",
                "text": "After greeting.You enter the second step: talk about what happened to user\n(hint: the friend "\
                "you see firework together)",
                "task_data": {
                    "respond_with_form":[
                        {
                            "type":"choices",
                            "question":"Template",
                            "choices":["Are you talk about ______?" ],
                        }, 
                        {"type":"text","question":"Input"}, 
                    ]
                  }
             })
        x = self.mturk_agent.act(timeout=self.opt["turn_timeout"])
        
        user.observe({
                    "id": "***User***",
                    "text": "Yes."  
                })
        
        user.observe({
                "id": "System",
                "text": "(hint: make an agrument )",
                "task_data": {
                    "respond_with_form":[
                        {
                            "type":"choices",
                            "question":"Template",
                            "choices":["What happend, did you ______?" ],
                        }, 
                        {"type":"text","question":"Input"}, 
                    ]
                  }
             })
        x = self.mturk_agent.act(timeout=self.opt["turn_timeout"])
        
        
        
        self.episodeDone = True

    def get_one_shared_memory(self):
        shared_memoryList = []
        for dialogue in self.shared_dialogues:
            shared_memoryList = dialogue["utterances"]
            break

        shared_memory = ""
        for utterance in shared_memoryList:
            shared_memory += utterance
        return "\n"+shared_memory



class MTurkMultiAgentDialogWorld(MTurkTaskWorld):

    def __init__(self, opt, agents=None, workers=None, shared=None):
        # Add passed in agents directly.
        self.agents = agents
        self.workers = workers
        self.acts = [None] * len(agents)
        self.episodeDone = False
        self.max_turns = opt.get("max_turns", 2)
        self.current_turns = 0
        self.send_task_data = opt.get("send_task_data", False)
        self.opt = opt
        self.agents[0].agent_id = "***User***"
        self.agents[1].agent_id = "***Wizard***"
        self.sent_first_message = False
        self.intention_chosen = False
        self.stage = 1
        self.user_instruction = False
        self.wizard_instruction = False
        
        
        self.stage_1 =[
         "[stage1]  Hey, how are you ?",
         "[stage1]  Hey, how's going on ?",
         "[stage1]  Hey, how's your day ?",
         "[Stage1]  Customize",
        ]
        
        self.stage_2 = [
          "[stage2]  Could you tell me more about ______ ?",
          "[stage2]  I wonder if you want to tell me about ______?",
          "[stage2]  Is ______ bothering you ?",
          "[stage2]  So you are feeling ______ , right?",
          "[stage2]  It can be hard to ______.",
          "[stage2]  I understand it means to be ______.",
          "[stage2]  Tell me why do you feel ______?",
          "[Stage2]  Customize",
        ]
        
        self.stage_3 = [
          "[stage3]  I recommand you try ______.",
          "[stage3]  Bye, feel free to ______.",
          "[stage3]  See ya, ______.",
          "[stage3]  Bye bye, hope the talk make you feel ______.",
          "[Stage3]  Customize",
        ]
        
        '''
        for idx, (agent,worker) in enumerate(zip(self.agents, self.workers)):
            if "monash" in worker.worker_name:
                self.wizard_agent = agent
                self.wizard_agent.agent_id = "Wizard"
            else:
                self.user_agent = agent
                self.user_agent.agent_id = "User"
        '''
        
        # read shared memory file 
        logger.info("read dialogues")
        data_path = os.path.join(get_root_dir(), "mephisto/tasks/shared-mem-chat/empatheticdialogues/empathetic-dialogues-train.json")
        fr = open(data_path,"r")
        self.shared_dialogues = json.load(fr)
        fr.close()
        self.shared_memory = self.get_one_shared_memory()
        logger.info(self.shared_memory)
        
        fw = open(data_path, "w", encoding="utf-8")
        json.dump(self.shared_dialogues, fw, ensure_ascii=False, indent=4)
        fw.close()
        logger.info("restore dialogues")
       
    def get_one_shared_memory(self):
        shared_memoryList = []
        for dialogue in self.shared_dialogues:
            # TODO least present_count
            if dialogue["present_count"] == 0:
                shared_memoryList = dialogue["utterances"]
                dialogue["present_count"] += 1
                break

        shared_memory = ""
        for utterance in shared_memoryList:
            shared_memory += utterance

        return "\n"+shared_memory
    
    
    def parley(self):
        """
        For each agent, get an observation of the last action each of the other agents
        took.
        Then take an action yourself.
        """
        acts = self.acts
        user = self.agents[0]
        wizard = self.agents[1]
        # user = self.user_agent
        # wizard = self.wizard_agent
        # Introduction and present the shared memory dialogue
        if not self.sent_first_message:
            user.observe({
                "id": "System",
                "text":"In this task, you will play the role as 'User'. Please read the following transcript carefully.",
            })
            user.observe({
                "id": "The topic happened before",
                "text": self.shared_memory,
            })
            wizard.observe({
                "id": "System",
                "text":"In this task, you will play the role as 'Wizard'. Please read the following transcript carefully.",
            })
            wizard.observe({
                "id": "The topic happened before",
                "text": self.shared_memory,
            })
            self.sent_first_message = True
        
        # Ask user choose one intention and situation
        if not self.intention_chosen:
            user.observe({
                "id": "System",
                "text": " You need to choose an intention for the new conversation. An intention is what is your goal to start this conversation, what problem do you want to solve in this conversation, or why do you start this conversation? Please select one from the following options. ",
                "task_data": {
                    "respond_with_form":[
                        {
                            "type":"choices",
                            "question":"Intention",
                            "choices":[
                                "I want to decrease my negative feelings.",
                                "I want to acquire knowledge of mental health.",
                                "I want to get some health suggestions.",
                                "I want to talk about movies, sport and music. ",
                                "I want to reduce my loneliness.",
                                "I just want to have some chat."],
                        }, 
                    ]
                    }
                })
            self.intention_chosen = True
            
            try:
                acts[0] = user.act(timeout=self.opt["turn_timeout"])
            except TypeError:
                acts[0] = user.act()  
            
            self.intention = acts[0]["task_data"]["form_responses"][0]["response"]
                
            user.observe({
                "id": "System",
                "text": "Please choose what kind of topic you want to talk about",
                "task_data": {
                    "respond_with_form":[
                        {
                            "type":"choices",
                            "question":"Situation",
                            "choices":[
                                "Continue the same topic with more details",
                                "Talking about a relevant topic",
                                "Begin a new topic",],
                        }, 
                    ]
                    }
                })
            
            try:
                acts[0] = user.act(timeout=self.opt["turn_timeout"])
            except TypeError:
                acts[0] = user.act()  
                
       # Enter the first stage(Greeting)
        if self.stage == 1:
            user.observe({
                "id": "System",
                "text": "Try to say hello to Ash",
                "task_data":{
                    "respond_with_form": [
                            {"type":"text","question":"Greeting"}, 
                        ]
                    }
                })
            
            try:
                acts[0] = user.act(timeout=self.opt["turn_timeout"])
            except TypeError:
                acts[0] = user.act()  
            
            uttr = acts[0]["task_data"]["form_responses"][0]["response"]
            acts[0].force_set("text",uttr)
            
            wizard.observe(validate(acts[0]))
            wizard.observe({
                "id": "System",
                "text": "Please choose one sentence to greet the user",
                "task_data": {
                    "respond_with_form":[
                        {
                            "type":"choices",
                            "question":"Greet",
                            "choices":self.stage_1
                        }, 
                        {"type":"text","question":"Input"}, 
                      
                    
                    ]
                  }
             })
            try:
                acts[1] = wizard.act(timeout=self.opt["turn_timeout"])
            except TypeError:
                acts[1] = wizard.act()  
            
            uttr = acts[1]["task_data"]["form_responses"][0]["response"]
            slot = acts[1]["task_data"]["form_responses"][1]["response"]
            if "Customize" in uttr:
                acts[1].force_set("text",slot)
            else:
                sentence = uttr.replace("______",slot)
                acts[1].force_set("text",sentence)
            
            user.observe(validate(acts[1]))
            self.stage = 2
        
        
       # Enter the second stage(Shared Memory)
        if self.stage == 2:
            if not self.user_instruction: 
                user.observe(
                        {
                            "id": "System",
                            "text": "According to your previous intention and situation setting,"\
                            "choose your emotion and start the conversation",
                            "task_data": {
                                "respond_with_form": [
                                    {
                                        "type":"choices",
                                        "question":"Emotion",
                                        "choices":["Happy","Sad","Anger","Disgust","Surprise","Fear","Lonely","Stress","Worried"],
                                    }, 
                                    {"type":"text","question":"Dialogue"},
                                ]
                            }
                        }
                    )
                try:
                    acts[0] = user.act(timeout=self.opt["turn_timeout"])
                except TypeError:
                    acts[0] = user.act()  

                self.user_instruction = True
            
            
            
            
            # get users' emotion and dialogue
            emotion = acts[0]["task_data"]["form_responses"][0]["response"]
            uttr = acts[0]["task_data"]["form_responses"][1]["response"]
            acts[0].force_set("text",uttr + " [" + emotion + "]")
 
           # Can be used to defined different template for different intention
            '''
            if not self.wizard_instruction:
            
                if self.intention == "Decrease negative feelings":
                    wizard.observe(validate(acts[0]))
                    wizard.observe({
                        "id": "System",
                        "text": "User want to decrease negative feeling,"\
                        "please choose one of the template and input phrase into the slots",
                         "task_data": {
                                    "respond_with_form": [
                                        {
                                            "type":"choices", 
                                            "question":"Template",
                                            "choices":self.stage_2 + self.stage_3
                                        },
                                        {"type":"text","question":"Input"}, 
                              
                                    ]
                                }

                    })
                elif self.intention == "Acquire knowledge of mental health":
                    wizard.observe(validate(acts[0]))
                    wizard.observe({
                        "id": "System",
                        "text": "User want to acquire knowledge of mental health,"\
                        "please choose one of the template and input phrase into the slots",
                         "task_data": {
                                   "respond_with_form": [
                                        {
                                            "type":"choices", 
                                            "question":"Template",
                                            "choices":self.stage_2 + self.stage_3
                                        },
                                        {"type":"text","question":"Input"}, 
                               
                                    ]

                                }

                    })
                elif self.intention == "Advocating health":
                    wizard.observe(validate(acts[0]))
                    wizard.observe({
                        "id": "System",
                        "text": "User want to advocating health,"\
                        "please choose one of the template and input phrase into the slots",
                         "task_data": {
                                    "respond_with_form": [
                                        {
                                            "type":"choices", 
                                            "question":"Template",
                                            "choices":self.stage_2 + self.stage_3
                                        },
                                        {"type":"text","question":"Input"}, 
                              
                                    ]
                                }

                    })
                elif self.intention == "Entertaining":
                    wizard.observe(validate(acts[0]))
                    wizard.observe({
                        "id": "System",
                        "text": "User want to have some entertaining,"\
                        "please choose one of the template and input phrase into the slots",
                         "task_data": {
                                    "respond_with_form": [
                                        {
                                            "type":"choices", 
                                            "question":"Template",
                                            "choices":self.stage_2 + self.stage_3
                                        },
                                        {"type":"text","question":"Input"}, 
                                   
                                    ]
                                }

                    })
                else:
                    wizard.observe(validate(acts[0]))
                    wizard.observe({
                        "id": "System",
                        "text": "User want to have a chat,"\
                        "please choose one of the template and input phrase into the slots",
                         "task_data": {
                                    "respond_with_form": [
                                        {
                                            "type":"choices", 
                                            "question":"Template",
                                            "choices":self.stage_2 + self.stage_3
                                        },
                                        {"type":"text","question":"Input"}, 
                                     
                                    ]
                                }

                    })
                self.wizard_instruction = True

            try:
                acts[1] = wizard.act(timeout=self.opt["turn_timeout"])
            except TypeError:
                acts[1] = wizard.act()  
        '''
            if not self.wizard_instruction:
                wizard.observe({
                        "id": "System",
                        "text": "User want to decrease negative feeling,"\
                        "please choose one of the template and input phrase into the slots",
                         "task_data": {
                                    "respond_with_form": [
                                        {
                                            "type":"choices", 
                                            "question":"Template",
                                            "choices":self.stage_2 + self.stage_3
                                        },
                                        {"type":"text","question":"Input"}, 

                                    ]
                                }

                    })
            try:
                acts[1] = wizard.act(timeout=self.opt["turn_timeout"])
            except TypeError:
                acts[1] = wizard.act() 

            uttr = acts[1]["task_data"]["form_responses"][0]["response"]
            slot = acts[1]["task_data"]["form_responses"][1]["response"]
          
            
            if "Customize" in uttr:
                acts[1].force_set("text",slot)
            else:
                sentence = uttr.replace("______",slot)
                acts[1].force_set("text",sentence)

            user.observe(validate(acts[1]))
            
            if "stage3" in uttr:
                self.stage = 3
            
            try:
                acts[0] = user.act(timeout=self.opt["turn_timeout"])
            except TypeError:
                acts[0] = user.act() 
            
            emotion = acts[0]["task_data"]["form_responses"][0]["response"]
            uttr = acts[0]["task_data"]["form_responses"][1]["response"]
            acts[0].force_set("text",uttr + " [" + emotion + "]")
            wizard.observe(validate(acts[0]))
        '''   
        # Enter stage_3, End the dialogue  
        if self.stage == 3:
            
            try:
                acts[1] = wizard.act(timeout=self.opt["turn_timeout"])
            except TypeError:
                acts[1] = wizard.act()  
            
            uttr = acts[1]["task_data"]["form_responses"][0]["response"]
            slot = acts[1]["task_data"]["form_responses"][1]["response"]
           
            
            if "Customize" in uttr:
                acts[1].force_set("text",slot)
            else:
                sentence = uttr.replace("______",slot)
                acts[1].force_set("text",sentence)

            user.observe(validate(acts[1]))
            
            try:
                acts[0] = user.act(timeout=self.opt["turn_timeout"])
            except TypeError:
                acts[0] = user.act()
            self.stage = 4
          '''  
            
          # End the dialogue
        if self.stage == 3:
            self.episodeDone = True
            for agent in self.agents:
                agent.observe(
                    {
                        "id": "System",
                        "text": "Please fill out the form to complete the chat:",
                        "task_data": {
                            "respond_with_form": [
                                {"type": "choices", 
                                 "question": "Please rank the performance of the person chat with you",
                                 "choices":["Very Good","Good","Netural","Poor","Very Poor"]},
                            ]
                        },
                    }
                )
                agent.act()  # Request a response
            for agent in self.agents:  # Ensure you get the response
                form_result = agent.act(timeout=self.opt["turn_timeout"])

    def prep_save_data(self, agent):
        """Process and return any additional data from this world you may want to store"""
        return {"example_key": "example_value"}

    def episode_done(self):
        return self.episodeDone

    def shutdown(self):
        """
        Shutdown all mturk agents in parallel, otherwise if one mturk agent is
        disconnected then it could prevent other mturk agents from completing.
        """
        global shutdown_agent

        def shutdown_agent(agent):
            try:
                agent.shutdown(timeout=None)
            except Exception:
                agent.shutdown()  

        Parallel(n_jobs=len(self.agents), backend="threading")(
            delayed(shutdown_agent)(agent) for agent in self.agents
        )


def make_onboarding_world(opt, agent):
    return MTurkMultiAgentDialogOnboardWorld(opt, agent)


def validate_onboarding(data):
    """Check the contents of the data to ensure they are valid"""
    print(f"Validating onboarding data {data}")
    return True


def make_world(opt, agents,workers):
    return MTurkMultiAgentDialogWorld(opt, agents,workers)


def get_world_params():
    return {"agent_count": 2}
