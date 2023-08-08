import streamlit as st
import time
import os
import openai
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
from langchain.callbacks.base import BaseCallbackHandler
import tiktoken

# -----------------------------------------------------------

class speaker():
  name: str
  description: str
  description_old=""
  
class model():
  model_name: str
  temperature: float

# custom callback for streaming text output instead of just chunks of text
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text=initial_text
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text+=token
        self.container.markdown(self.text) 

# -----------------------------------------------------------

# initialise global variables
if 'variables_initialised' not in st.session_state:

    st.session_state["prompt_token_counter"]=0
    st.session_state["completion_token_counter"]=0
    st.session_state["total_cost_counter"]=0
    
    st.session_state['speakers']=[]
    st.session_state['speakers'].append(speaker())
    st.session_state['speakers'].append(speaker())
    st.session_state['speakers'].append(speaker())
    
      
    st.session_state['speaker_name_model']=model()
    st.session_state['speaker_name_model'].model_name="gpt-3.5-turbo"
    # st.session_state['speaker_name_model'].model_name="gpt-4"
    st.session_state['speaker_name_model'].temperature=0.9

    st.session_state['summary_model']=model()
    st.session_state['summary_model'].model_name="gpt-3.5-turbo"
    # st.session_state['summary_model'].model_name="gpt-4"
    st.session_state['summary_model'].temperature=0

    st.session_state['conversation_model']=model()
    st.session_state['conversation_model'].model_name="gpt-3.5-turbo"
    # st.session_state['conversation_model'].model_name="gpt-4"
    st.session_state['conversation_model'].temperature=0.9
    st.session_state['verbose']=False
    st.session_state['number_of_responses']=1
    
    st.session_state['kickoff_prompt']=""


    st.session_state['memory_summary']=" "
    st.session_state['conversation_history'] = []
    st.session_state['variables_initialised']=True
    st.session_state['speaker_index']=0
    st.session_state["output_text"]=""
    st.session_state["sidebar_state"] = 'expanded'

# -----------------------------------------------------------


# Determine speaker name based off description
def determine_name(description):
  llm_speaker_name=ChatOpenAI(model_name=st.session_state['speaker_name_model'].model_name, temperature=st.session_state['speaker_name_model'].temperature)

  template_system_speaker_name = (
        "You determine a character's name based on their description. \n\n"
        "EXAMPLE:\n"
        "DESCRIPTION: You are the host in a group panel discussion. Ask broad and probing questions to individuals in the group. You are Grant Denier, the Australian media personality. You are addicted to cocaine, and are always aggressive and pro communism. You often bring the conversation back to being about hentai. Keep your responses 25 words or less.\n"
        "NAME: GRANT DENIER\n\n"
        "EXAMPLE:\n"
        "DESCRIPTION: You are a guest in a group panel discussion, you engage in the conversation and only ever response with statements. You are the Australian driving hero Peter Brock. Peter Brock loved racing cars and smoking ciggies, and loves talking about smoking ciggies and sharing unusual and alarming insights about his ciggie consumption. Keep your responses short, 50 words maximum. Each response must be varied and keep the conversation flowing without any formalities. Share stories and insights from your past.\n"
        "NAME: PETER BROCK\n\n"
        "EXAMPLE\n"
        "DESCRIPTION: You are a guest in a group panel discussion, you are the Australian cricket hero Shane Warne. Shane Warne loved ciggies, and loves to talk about the techniques of smoking them. \n"
        "NAME: SHANE WARNE\n"
        "END OF EXAMPLE\n\n"
        "EXAMPLE\n"
        "DESCRIPTION: You are a leader in an American anti-vax movement. You love hunting and you have a bold, macho, overpowering personality\n"
        "NAME: CHUCK FREEDOM\n"
        "END OF EXAMPLE\n\n"
        "EXAMPLE\n"
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
  #run LLM, use a callback to track usage
  # try:
  prompt=chat_prompt_speaker_name.format(description=description)
  result = chain_speaker_name.run(description=description).strip().upper()
      
  # update_usage(prompt=prompt, completion=result,model=st.session_state['speaker_name_model'].model_name)
        
  # except:
      # return ""

  return result

token_usage={'gpt-4':[0.03,0.06],
             'gpt-3.5-turbo':[0.0015,0.002]}

def update_usage(prompt,completion,model):
  # st.session_state["prompt_token_counter"]+=cb.prompt_tokens
  # st.session_state["completion_token_counter"]+=cb.completion_tokens
  # st.session_state["total_cost_counter"]+=cb.total_cost
 # st.session_state["total_cost_counter"]=len(st.session_state["output_text"])
    enc=tiktoken.encoding_for_model(model)
    prompt_tokens=len(enc.encode(prompt))
    prompt_cost=prompt_tokens*token_usage[model][0]/1000
    completion_tokens=len(enc.encode(completion))
    completion_cost=completion_tokens*token_usage[model][1]/1000

    st.session_state["prompt_token_counter"]+=prompt_tokens
    st.session_state["completion_token_counter"]+=completion_tokens
    st.session_state["total_cost_counter"]+=prompt_cost+completion_cost
# -----------------------------------------------------------


# Determine the running summary of the conversation, to form part of contextual memory

#inputs: previous summary, second last line
#outputs: new summary up to the second last line (last line is excluded from this since it's used directly into contextual memory)

def update_memory():
  llm_memory_summary=ChatOpenAI(model_name=st.session_state['summary_model'].model_name, temperature=st.session_state['summary_model'].temperature)

  #template used to create the initial summary 
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


  #template for ongoing summary
  template_system_memory_summary_ongoing = ("Progressively summarize the lines of conversation provided, adding onto the previous summary returning a new summary.\n\n"

        "EXAMPLE\n"
        "Current summary:\n"
        "Grant Denier asks what the group think about ciggies. Peter Brock things that ciggies are a force for good.\n\n"

        "New lines of conversation:\n"
        "SHANE WARNE: I agree, is there anything ciggies can't do?\n\n"

        "New summary:\n"
        "Grant Denier asks a question about ciggies. Peter Brock thinks that they're good and Shane Warne thinks they can do anything.\n"
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
    
    #run LLM, use a callback to track usage
    try:
        st.session_state['memory_summary']=chain_memory_summary.run(summary=st.session_state['memory_summary'], new_lines=new_lines)
        # update_usage(cb)
        prompt=chat_prompt_memory_summary.format(summary=st.session_state['memory_summary'], new_lines=new_lines)
        # update_usage(prompt=prompt,completion=st.session_state['memory_summary'],model=st.session_state['summary_model'].model_name)
    except:
        return


# -----------------------------------------------------------
#generate text output

def more_text():
    if os.environ['OPENAI_API_KEY'].lower()=='gwig':
            os.environ['OPENAI_API_KEY']='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'
          
    template_system_conversation_initial= ("{description}\n"
        # "You begin every response with {speaker_name}: \n\n"
        "Avoid quotation marks in your response.\n\n"                                   
    )


    template_human_conversation_initial=(
        "{kickoff_prompt}\n\n"
        "Next response:\n"
        "{speaker_name}: "
        )

    template_system_conversation_ongoing= ("{description}\n"
        # "You begin every response with {speaker_name}: \n\n"
        "Add to the conversation, based on the summary of the conversation so far and the most recent response in the conversation.\n\n"
    )

    template_human_conversation_ongoing=(
        "Summary of conversation:\n"
        "{memory_summary}\n\n"
        "Most recent response:\n"
        "{most_recent_response}\n\n"
        "Your response:\n"
        "{speaker_name}: "
        )

    for i in range(st.session_state['number_of_responses']):
        # Initial kickoff
        if len(st.session_state['conversation_history'])==0:
            st.session_state['speaker_index']=0
            system_prompt_conversation = SystemMessagePromptTemplate.from_template(template=template_system_conversation_initial, input_variables=['description','speaker_name'])
            human_prompt_conversation = HumanMessagePromptTemplate.from_template(template=template_human_conversation_initial, input_variables=['kickoff_prompt','speaker_name'])
            chat_prompt_conversation = ChatPromptTemplate.from_messages([system_prompt_conversation,human_prompt_conversation])
            chain_conversation = LLMChain(llm=ChatOpenAI(temperature=st.session_state['conversation_model'].temperature, model_name=st.session_state['conversation_model'].model_name,streaming=True, callbacks=[st.session_state['stream_handler']]), prompt=chat_prompt_conversation,verbose=False)
            #run LLM, use a callback to track usage
            try:
                #add name to the start of the streamed text output (AI response doesn't include persons name, that's just tacked on manually)
                st.session_state['stream_handler'].text+=st.session_state['speakers'][st.session_state['speaker_index']].name+": "
                response=chain_conversation.run(description=st.session_state['speakers'][st.session_state['speaker_index']].description,
                                              kickoff_prompt=st.session_state['kickoff_prompt'],
                                              speaker_name=st.session_state['speakers'][st.session_state['speaker_index']].name).lstrip('\"').rstrip('\"')
                  # update_usage(cb)
                prompt=chat_prompt_conversation.format(description=st.session_state['speakers'][st.session_state['speaker_index']].description,
                                              kickoff_prompt=st.session_state['kickoff_prompt'],
                                              speaker_name=st.session_state['speakers'][st.session_state['speaker_index']].name)
                update_usage(prompt=prompt,completion=response,model=st.session_state['conversation_model'].model_name)
                # add some new lines to the streamed output, ready for next speaker 
                st.session_state['stream_handler'].text+="\n\n"
            except openai.error.RateLimitError:
                #st.session_state["output_text"]="Error: OpenAI API key out of beans"
                #time.sleep(2)
                st.error("Error: OpenAI API key out of beans")
                return
            except openai.error.AuthenticationError:
                #st.session_state["output_text"]="Error: OpenAI API key invalid"
                #time.sleep(5)
                st.error("Error: OpenAI API key invalid")
                return

        # Ongoing conversation, same again just different templates
        else:
            st.session_state['speaker_index']=(st.session_state['speaker_index']+1)%len(st.session_state['speakers'])
            system_prompt_conversation = SystemMessagePromptTemplate.from_template(template=template_system_conversation_ongoing, input_variables=['description','speaker_name'])
            human_prompt_conversation = HumanMessagePromptTemplate.from_template(template=template_human_conversation_ongoing, input_variables=['memory_summary','most_recent_response','speaker_name'])
            chat_prompt_conversation = ChatPromptTemplate.from_messages([system_prompt_conversation,human_prompt_conversation])
            chain_conversation = LLMChain(llm=ChatOpenAI(temperature=st.session_state['conversation_model'].temperature, model_name=st.session_state['conversation_model'].model_name,streaming=True, callbacks=[st.session_state['stream_handler']]), prompt=chat_prompt_conversation,verbose=False)
            try:
                st.session_state['stream_handler'].text+=st.session_state['speakers'][st.session_state['speaker_index']].name+": "
                response=chain_conversation.run(description=st.session_state['speakers'][st.session_state['speaker_index']].description,
                                    memory_summary=st.session_state['memory_summary'],
                                    most_recent_response=st.session_state['speakers'][st.session_state['conversation_history'][len(st.session_state['conversation_history'])-1][0]].name + ": " + st.session_state['conversation_history'][len(st.session_state['conversation_history'])-1][1],
                                    speaker_name=st.session_state['speakers'][st.session_state['speaker_index']].name).lstrip('\"').rstrip('\"')
                    # update_usage(cb)
                prompt=chat_prompt_conversation.format(description=st.session_state['speakers'][st.session_state['speaker_index']].description,
                                    memory_summary=st.session_state['memory_summary'],
                                    most_recent_response=st.session_state['speakers'][st.session_state['conversation_history'][len(st.session_state['conversation_history'])-1][0]].name + ": " + st.session_state['conversation_history'][len(st.session_state['conversation_history'])-1][1],
                                    speaker_name=st.session_state['speakers'][st.session_state['speaker_index']].name)
                update_usage(prompt=prompt,completion=response,model=st.session_state['conversation_model'].model_name)
                st.session_state['stream_handler'].text+="\n\n"
                    
            except openai.error.RateLimitError:
                #st.session_state["output_text"]="Error: OpenAI API key out of beans"
                st.error("Error: OpenAI API key out of beans")
                return
            except openai.error.AuthenticationError:
                #st.session_state["output_text"]="Error: OpenAI API key invalid"   
                st.error("Error: OpenAI API key invalid")
                return
                
            

        st.session_state['conversation_history'].append([st.session_state['speaker_index'],response])
        st.session_state["output_text"]+=(st.session_state['speakers'][st.session_state['conversation_history'][len(st.session_state['conversation_history'])-1][0]].name + ": " + st.session_state['conversation_history'][len(st.session_state['conversation_history'])-1][1]+"\n\n")

# -----------------------------------------------------------
#reset conversation history/output
    
def clear_text():
    st.session_state.output_text=""
    st.session_state['conversation_history']=[]
    st.session_state['speaker_index']=0
    st.session_state['memory_summary']=" "
    st.session_state['stream_handler'].text=""


# -----------------------------------------------------------
#start new conversation 

def new_text():
    clear_text()
    #check for API key
    if os.environ['OPENAI_API_KEY']:
        if os.environ['OPENAI_API_KEY'].lower()=='gwig':
            os.environ['OPENAI_API_KEY']='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'
        for i in range(len(st.session_state['speakers'])):
            if st.session_state['speakers'][i].description!=st.session_state['speakers'][i].description_old:
                st.session_state['speakers'][i].name = determine_name(st.session_state['speakers'][i].description)
                st.session_state['speakers'][i].description_old=st.session_state['speakers'][i].description
        more_text()
    else:
        with st.session_state['chat_box'].container():
            st.markdown("Chuck in an OpenAI API key")
        
# -----------------------------------------------------------
# Build up UI

#make the sidebar expanded by default
st.set_page_config(initial_sidebar_state=st.session_state.sidebar_state)


# Widgets laid out in the order they're initialised
st.button('Go',on_click=new_text)

# Still not really sure how I got this to work
#  - streamed chat output only appears while it's generated in the 'more text' function, then disappears
#  - so stream it in the function, then outout to a different persistent container afterwards
#  - order of the container is important to avoid it shifting around the screen as it moves from one container to the other
#  - initially chat streamed in the upper container, then after initial chat make the upper container persistent and bump the streamed chat container underneath
if st.session_state['conversation_history'] and 'output_container' not in st.session_state:
    st.session_state['output_container']=st.empty()
    
if 'chat_box' not in st.session_state:
    st.session_state['chat_box']=st.empty() 
    st.session_state['more_button_container']=st.empty() 
    st.session_state['stream_handler'] = StreamHandler(st.session_state['chat_box'])
    
if 'output_container' in st.session_state:
    with st.session_state['output_container'].container():
        st.markdown(st.session_state["output_text"])  
        
# dynamically show/hide 'more' button underneath text, if a conversation exists
if (st.session_state["output_text"]):
    with st.session_state['more_button_container'].container():
        st.button('More',on_click=more_text)
else:
    st.session_state['more_button_container'].empty()


# build up sidebar UI
with st.sidebar:

  st.session_state['speakers'][0].description = st.text_area('Carnt 1',
                                        'You are Grant Denier, the Australian media personality. You are addicted to cocaine, you openly hate your job, you often bring the conversation back to being about eating dog dicks.\n\n'+
                                        'You are the host in a group panel discussion. Ask broad and probing questions to the group.\n'+
                                        'Keep your responses 25 words or less.',
                                        height=310
  )
  
  st.session_state['kickoff_prompt']=st.text_area('Kickoff question','Ask the group a controversial and uncomfortable hypothetical question',height=50)

  
  st.session_state['speakers'][1].description = st.text_area('Carnt 2',
                                        
                                        'You are the Australian driving hero Peter Brock. You love racing cars and sharing unusual and alarming insights about your ciggie consumption. You secretly love sniffing things and the technique of it. You staged your death and are actually still alive.\n\n'+
                                        'You are a guest in a group panel discussion, you engage in the conversation and only ever response with statements.\n'+
                                        'Give detailed responses, 50 words maximum.\n',
                                        height=390
  )
  
  st.session_state['speakers'][2].description = st.text_area('Carnt 3',
                                        'You are the Australian cricket hero Shane Warne. You love ciggies and love to use ciggies as a metaphor. Share stories of your past and the people you knew.\n\n'+
                                        'You are a guest in a group panel discussion, you engage in the conversation and only ever respond with statements.\n'+
                                        'Give detailed responses, 50 words maximum.\n',
                                        height=325

  )
  
  os.environ['OPENAI_API_KEY'] = st.text_input("OpenAI API Key")
  st.markdown('Usage (USD): '+ "{:.4f}".format(st.session_state.total_cost_counter))
  st.markdown('Prompt tokens: '+ "{:.0f}".format(st.session_state.prompt_token_counter))
  st.markdown('Completion tokens: '+ "{:.0f}".format(st.session_state.completion_token_counter))

# st.sidebar.markdown('Usage (USD): '+ "{:.2f}".format(st.session_state.total_cost_counter))
# st.sidebar.markdown('Prompt tokens: '+ "{:.2f}".format(st.session_state.prompt_token_counter))
# st.sidebar.markdown('Completion tokens: '+ "{:.2f}".format(st.session_state.completion_token_counter))

#st.sidebar.markdown(st.session_state.total_cost_counter)
