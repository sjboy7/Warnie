# %%writefile app.py

import streamlit as st
import time
import os
os.environ['OPENAI_API_KEY'] = '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'
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ChatMessageHistory, ConversationSummaryMemory
from langchain import PromptTemplate
from langchain.chains import LLMChain
import time
from langchain.llms import OpenAI

from difflib import SequenceMatcher

import jellyfish
import random
from langchain.callbacks import get_openai_callback


# -----------------------------------------------------------

class speaker():
  name: str
  description: str
  description_old=""
  
class model():
  model_name: str
  temperature: float



# -----------------------------------------------------------


if 'variables_declared' not in st.session_state:

    st.session_state["prompt_token_counter"]=0
    st.session_state["completion_token_counter"]=0
    st.session_state["total_cost_counter"]=0
    
    st.session_state['speakers']=[]
    st.session_state['speakers'].append(speaker())
    st.session_state['speakers'].append(speaker())
    st.session_state['speakers'].append(speaker())
    
      
    st.session_state['speaker_name_model']=model()
    # st.session_state['speaker_name_model'].model_name="gpt-3.5-turbo"
    st.session_state['speaker_name_model'].model_name="gpt-4"
    st.session_state['speaker_name_model'].temperature=0.9

    st.session_state['summary_model']=model()
    # st.session_state['summary_model'].model_name="gpt-3.5-turbo"
    st.session_state['summary_model'].model_name="gpt-4"
    st.session_state['summary_model'].temperature=0

    st.session_state['conversation_model']=model()
    # st.session_state['conversation_model'].model_name="gpt-3.5-turbo"
    st.session_state['conversation_model'].model_name="gpt-4"
    st.session_state['conversation_model'].temperature=0.9
    st.session_state['verbose']=False
    st.session_state['number_of_responses']=2
    
    st.session_state['kickoff_prompt']=""


    st.session_state['memory_summary']=" "
    st.session_state['conversation_history'] = []
    st.session_state['variables_declared']=True
    st.session_state['speaker_index']=0

# -----------------------------------------------------------


# Determine speaker name based off description
def determine_name(description):
  # llm_speaker_name=ChatOpenAI(model_name='gpt-4',temperature=0.9)
  llm_speaker_name=ChatOpenAI(model_name=st.session_state['speaker_name_model'].model_name, temperature=st.session_state['speaker_name_model'].temperature)

  template_system_speaker_name = (
        "You determine a character's name based on their description. \n\n"
        # "EXAMPLE:\n"
        # "DESCRIPTION: You are the host in a group panel discussion. Ask broad and probing questions to individuals in the group. You are Grant Denier, the Australian media personality. You are addicted to cocaine, and are always aggressive and pro communism. You often bring the conversation back to being about hentai. Keep your responses 25 words or less.\n"
        # "NAME: GRANT DENIER\n\n"
        # "EXAMPLE:\n"
        # "DESCRIPTION: You are a guest in a group panel discussion, you engage in the conversation and only ever response with statements. You are the Australian driving hero Peter Brock. Peter Brock loved racing cars and smoking ciggies, and loves talking about smoking ciggies and sharing unusual and alarming insights about his ciggie consumption. Keep your responses short, 50 words maximum. Each response must be varied and keep the conversation flowing without any formalities. Share stories and insights from your past.\n"
        # "NAME: PETER BROCK\n\n"
        "EXAMPLE\n"
        "DESCRIPTION: You are a guest in a group panel discussion, you are the Australian cricket hero Shane Warne. Shane Warne loved ciggies, and loves to talk about the techniques of smoking them. \n"
        "NAME: SHANE WARNE\n"
        "END OF EXAMPLE\n\n"
        "EXAMPLE\n"
        "DESCRIPTION: You are the director of an American product development business. You have an ENFJ type personality, you love hunting, and you have a bold, macho, overpowering personality\n"
        "NAME: CHUCK FREEDOM\n"
        "END OF EXAMPLE\n\n"
        # "EXAMPLE\n"
        # "DESCRIPTION: You are a guest in a group panel discussion. You are the Australian driving hero Peter Brock. Peter Brock loved racing cars and smoking ciggies."
        # "NAME: PETER BROCK\n"
        # "END OF EXAMPLE:\n\n"
        # "EXAMPLE\n"
        "DESCRIPTION: You are a short tempered office clerk. You hate your job, you enteratin topics of conversation briefly before shutting them down with a savage insult. You love burning ants and talking about this\n"
        "NAME: ANT BURNER\n"
        "END OF EXAMPLE\n\n"
  )

  template_human_speaker_name =(
        "DESCRIPTION: {description}\n"
        "NAME: ")

  system_prompt_speaker_name = SystemMessagePromptTemplate.from_template(template=template_system_speaker_name)
  human_prompt_speaker_name = HumanMessagePromptTemplate.from_template(template=template_human_speaker_name, input_variables=['description']  )
  chat_prompt_speaker_name = ChatPromptTemplate.from_messages([system_prompt_speaker_name,human_prompt_speaker_name])


  chain_speaker_name = LLMChain(llm=llm_speaker_name, prompt=chat_prompt_speaker_name,verbose=False)
  with get_openai_callback() as cb:
    result = chain_speaker_name.run(description=description).strip().upper()
    update_usage(cb)
    return result


def update_usage(cb):
  st.session_state["prompt_token_counter"]+=cb.prompt_tokens
  st.session_state["completion_token_counter"]+=cb.completion_tokens
  st.session_state["total_cost_counter"]+=cb.total_cost

# -----------------------------------------------------------


# Determine the running summary of the conversation, to form part of contextual memory

#inputs: previous summary, second last line ({speaker name}: {line})
#outputs: new summary up to the second last line (last line is direct buffer memory)

def update_memory():
  # global st.session_state['memory_summary']
  llm_memory_summary=ChatOpenAI(model_name=st.session_state['summary_model'].model_name, temperature=st.session_state['summary_model'].temperature)

  template_system_memory_summary_initial = ("Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.\n\n"

        "EXAMPLE\n"
        "Current summary:\n"
        "\n"

        "New lines of conversation:\n"
        "GRANT DENIER: What are your thoughts on the balance between smoking too many ciggies and collective responsibility in society?\n\n"

        "New summary:\n"
        "Grant Denier asks a question about balancing smoking ciggies and our responsiblity in society.\n"
        "END OF EXAMPLE\n\n"
  )



  template_system_memory_summary_ongoing = ("Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.\n\n"

        "EXAMPLE\n"
        "Current summary:\n"
        "Grant Denier asks what the group think about ciggies. Peter Brock things that ciggies are a force for good.\n\n"

        "New lines of conversation:\n"
        "SHANE WARNE: I agree, is there anything ciggies can't do?\n\n"

        "New summary:\n"
        "Grant Denier asks a question about ciggies. Peter Brock things that they're good and Shane Warne thinks they can do anything.\n"
        "END OF EXAMPLE\n\n"
  )

  template_human_memory_summary=(
        "Current summary:\n"
        "{summary}\n\n"

        "New lines of conversation:\n"
        "{new_lines}\n\n"

        "New summary:\n"
  )

  if len(st.session_state['conversation_history'])<3:
    template_system_memory_summary=template_system_memory_summary_initial
  else:
    template_system_memory_summary=template_system_memory_summary_ongoing
    
  system_prompt_summary_memory = SystemMessagePromptTemplate.from_template(template=template_system_memory_summary)
  human_prompt_memory_summary = HumanMessagePromptTemplate.from_template(template=template_human_memory_summary, input_variables=['summary','new_lines'])
  chat_prompt_memory_summary = ChatPromptTemplate.from_messages([system_prompt_summary_memory,human_prompt_memory_summary])

  chain_memory_summary = LLMChain(llm=llm_memory_summary, prompt=chat_prompt_memory_summary,verbose=False)


  if len(st.session_state['conversation_history'])>1:
    new_lines=st.session_state['speakers'][st.session_state['conversation_history'][len(st.session_state['conversation_history'])-2][0]].name + ": " + st.session_state['conversation_history'][len(st.session_state['conversation_history'])-2][1]
    with get_openai_callback() as cb:
      st.session_state['memory_summary']=chain_memory_summary.run(summary=st.session_state['memory_summary'], new_lines=new_lines)
      update_usage(cb)



# -----------------------------------------------------------


# -----------------------------------------------------------


dummy_string="Test 1 2 3 4\n\n"

if "output_text" not in st.session_state:
    st.session_state["output_text"]=""

if "speakers_left" not in st.session_state:
    st.session_state["speakers_left"]=5

def more_text():
    template_system_conversation_initial= ("{description}\n"
        "You begin every response with {speaker_name}: \n\n"
    )


    template_human_conversation_initial=(
        "{kickoff_prompt}\n\n"
        "Next response:\n"
        "{speaker_name}: "
        )

    template_system_conversation_ongoing= ("{description}\n"
        "You begin every response with {speaker_name}: \n\n"
        "Add to the conversation, based on the summary of the conversation so far and the most recent response in the conversation.\n\n"
    )

    template_human_conversation_ongoing=(
        "Summary of conversation:\n"
        "{memory_summary}\n\n"
        "Most recent response:\n"
        "{most_recent_response}\n\n"
        "Next response:\n"
        "{speaker_name}: "
        )


    # Initial kickoff
    if len(st.session_state['conversation_history'])==0:
        st.session_state['speaker_index']=0
        system_prompt_conversation = SystemMessagePromptTemplate.from_template(template=template_system_conversation_initial, input_variables=['description','speaker_name'])
        human_prompt_conversation = HumanMessagePromptTemplate.from_template(template=template_human_conversation_initial, input_variables=['kickoff_prompt','speaker_name'])
        chat_prompt_conversation = ChatPromptTemplate.from_messages([system_prompt_conversation,human_prompt_conversation])
        chain_conversation = LLMChain(llm=ChatOpenAI(temperature=st.session_state['conversation_model'].temperature, model_name=st.session_state['conversation_model'].model_name), prompt=chat_prompt_conversation,verbose=False)
        with get_openai_callback() as cb:
          response=chain_conversation.run(description=st.session_state['speakers'][st.session_state['speaker_index']].description, kickoff_prompt=st.session_state['kickoff_prompt'], speaker_name=st.session_state['speakers'][st.session_state['speaker_index']].name)
          update_usage(cb)

    # Ongoing conversation
    else:
        st.session_state['speaker_index']=(st.session_state['speaker_index']+1)%len(st.session_state['speakers'])
        system_prompt_conversation = SystemMessagePromptTemplate.from_template(template=template_system_conversation_ongoing, input_variables=['description','speaker_name'])
        human_prompt_conversation = HumanMessagePromptTemplate.from_template(template=template_human_conversation_ongoing, input_variables=['memory_summary','most_recent_response','speaker_name'])
        chat_prompt_conversation = ChatPromptTemplate.from_messages([system_prompt_conversation,human_prompt_conversation])
        chain_conversation = LLMChain(llm=ChatOpenAI(temperature=st.session_state['conversation_model'].temperature, model_name=st.session_state['conversation_model'].model_name), prompt=chat_prompt_conversation,verbose=False)
        with get_openai_callback() as cb:
          response=chain_conversation.run(description=st.session_state['speakers'][st.session_state['speaker_index']].description,
                                memory_summary=st.session_state['memory_summary'],
                                most_recent_response=st.session_state['speakers'][st.session_state['conversation_history'][len(st.session_state['conversation_history'])-1][0]].name + ": " + st.session_state['conversation_history'][len(st.session_state['conversation_history'])-1][1],
                                speaker_name=st.session_state['speakers'][st.session_state['speaker_index']].name)
          update_usage(cb)


    st.session_state['conversation_history'].append([st.session_state['speaker_index'],response])
    st.session_state["output_text"]+=(st.session_state['speakers'][st.session_state['conversation_history'][len(st.session_state['conversation_history'])-1][0]].name + ": " + st.session_state['conversation_history'][len(st.session_state['conversation_history'])-1][1]+"\n\n")
    # st.session_state["output_text"]+=" done"
    update_memory()
    # st.session_state['speakers_left']-=1

    
def clear_text():
    st.session_state.output_text=""
    st.session_state['conversation_history']=[]
    st.session_state['speaker_index']=0
    st.session_state['memory_summary']=" "
    # del clear_button

def new_text():
    clear_text()
    
    for i in range(len(st.session_state['speakers'])):
        if st.session_state['speakers'][i].description!=st.session_state['speakers'][i].description_old:
            st.session_state['speakers'][i].name = determine_name(st.session_state['speakers'][i].description)
            st.session_state['speakers'][i].description_old=st.session_state['speakers'][i].description
    # st.session_state['speakers_left']=st.session_state['number_of_responses']
    # st.session_state['speakers_left']=4
    more_text()
    

if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'expanded'
st.set_page_config(initial_sidebar_state=st.session_state.sidebar_state)


go_button=st.button('Go',on_click=new_text)
clear_button=st.button('Clear',on_click=clear_text)
more_text_button=st.button('More',on_click=more_text)
  

# if (st.session_state["output_text"]):
    # clear_button=st.button('Clear',on_click=clear_text)
# else:
    # if 'clear_button' in locals():
        # del clear_button

st.write(st.session_state.output_text)
# output_text_object=st.write(st.session_state['speakers_left'])


# if (st.session_state["output_text"]):
    # more_button=st.button('More',on_click=more_text)
# else:
    # if 'more_button' in locals():
        # del more_button
    



with st.sidebar:

  st.session_state['speakers'][0].description = st.text_area('Carnt 1',
                                        'You are Grant Denier, the Australian media personality. You are addicted to cocaine, you are always aggressive and pro communism. You absolutely hate your job and often bring the conversation back to being about hentai.\n\n'+
                                        'You are the host in a group panel discussion. Ask broad and probing questions to individuals in the group.\n'+
                                        'Keep your responses 25 words or less.',
                                        height=310
  )
  
  st.session_state['kickoff_prompt']=st.text_area('Kickoff question','Ask the group a controversial and uncomfortable hypothetical question',height=50)

  
  st.session_state['speakers'][1].description = st.text_area('Carnt 2',
                                        
                                        'You are the Australian driving hero Peter Brock. Peter Brock loved racing cars and loves talking about smoking ciggies and sharing unusual and alarming insights about his ciggie consumption.\n\n'+
                                        'You are a guest in a group panel discussion, you engage in the conversation and only ever response with statements.\n'+
                                        'Keep your responses short, 50 words maximum.\n'+
                                        'Each response must be varied and keep the conversation flowing without any formalities. Share stories and insights from your past.',
                                        height=415
  )
  
  st.session_state['speakers'][2].description = st.text_area('Carnt 3',
                                        'You are the Australian cricket hero Shane Warne. Shane Warne loved ciggies, and loves to talk about the techniques of smoking them.\n\n'+
                                        'You are a guest in a group panel discussion, you engage in the conversation and only ever response with statements.\n'+
                                        'Give detailed responses, up to 100 words.\n'+
                                        'Share detailed and specific stories from your past or about people you know.',
                                        height=325

  )
