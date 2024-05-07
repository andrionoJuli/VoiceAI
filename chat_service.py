from langchain.chains import LLMChain  # To be replaced with new chains method once alternatives are available
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory


class Chat:
    def __init__(
            self,
            prompt,
    ):
        self.template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=prompt
                ),  # The persistent system prompt
                MessagesPlaceholder(
                    variable_name="chat_history"
                ),  # Where the memory will be stored.
                HumanMessagePromptTemplate.from_template(
                    "{input}"
                ),  # Where the human input will be injected.
            ]
        )

        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        self.model = Ollama(model="llama3")

        self.chain = LLMChain(
            llm=self.model,
            prompt=self.template,
            verbose=True,
            memory=self.memory
        )

    def __call__(self, text, *args, **kwargs):
        return self.chain.predict(input=text)
