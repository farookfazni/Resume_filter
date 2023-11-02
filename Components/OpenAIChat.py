
# def handle_user_input(question):
#     response = st.session_state.conversation({'question':question})
#     st.session_state.chat_history = response('chat_history')

#     for i,message in enumerate(st.session_state.chat_history):
#         if i % 2 == 0:
#             st.write(user_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
#         else:
#             st.write(bot_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)

# def get_conversation_chain(vector_store):
#     llm = ChatOpenAI()
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vector_store.as_retriever(),
#         memory = memory
#     )
#     return conversation_chain