# %%writefile app.py

import streamlit as st
import time

class speaker():
  name: str
  description: str
  description_old=""

speakers=[]

speakers.append(speaker())
speakers.append(speaker())
speakers.append(speaker())

dummy_string="Test 1 2 3 4\n\n"

if "output_text" not in st.session_state:
    st.session_state["output_text"]=""

def more_text():
    st.session_state.output_text+=dummy_string

    
def clear_text():
    st.session_state.output_text=""
    # del clear_button

def new_text():
    clear_text()
    
    for i in range(len(speakers)):
        if speakers[i].description!=speakers[i].description_old:
            pass
    more_text()


if 'sidebar_state' not in st.session_state:
    st.session_state.sidebar_state = 'expanded'
st.set_page_config(initial_sidebar_state=st.session_state.sidebar_state)


go_button=st.button('Go',on_click=new_text)



   

if (st.session_state["output_text"]):
    clear_button=st.button('Clear',on_click=clear_text)
else:
    if 'clear_button' in locals():
        del clear_button



st.write(st.session_state.output_text)

if (st.session_state["output_text"]):
    more_button=st.button('More',on_click=more_text)
else:
    if 'more_button' in locals():
        del more_button
    



with st.sidebar:

  speakers[0].description = st.text_area('Carnt 1',
                                        'You are Grant Denier, the Australian media personality. You are addicted to cocaine, you are always aggressive and pro communism. You absolutely hate your job and often bring the conversation back to being about hentai.\n\n'+
                                        'You are the host in a group panel discussion. Ask broad and probing questions to individuals in the group.\n'+
                                        'Keep your responses 25 words or less.',
                                        height=310
  )
  
  kickoff_prompt=st.text_area('Kickoff question','Ask the group a controversial and uncomfortable hypothetical question',height=50)

  
  speakers[1].description = st.text_area('Carnt 2',
                                        
                                        'You are the Australian driving hero Peter Brock. Peter Brock loved racing cars and loves talking about smoking ciggies and sharing unusual and alarming insights about his ciggie consumption.\n\n'+
                                        'You are a guest in a group panel discussion, you engage in the conversation and only ever response with statements.\n'+
                                        'Keep your responses short, 50 words maximum.\n'+
                                        'Each response must be varied and keep the conversation flowing without any formalities. Share stories and insights from your past.',
                                        height=415
  )
  
  speakers[2].description = st.text_area('Carnt 3',
                                        'You are the Australian cricket hero Shane Warne. Shane Warne loved ciggies, and loves to talk about the techniques of smoking them.\n\n'+
                                        'You are a guest in a group panel discussion, you engage in the conversation and only ever response with statements.\n'+
                                        'Give detailed responses, up to 100 words.\n'+
                                        'Share detailed and specific stories from your past or about people you know.',
                                        height=325

  )






        

