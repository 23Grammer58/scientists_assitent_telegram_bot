import json
import os
import sys

from pathlib import Path

from collections import defaultdict

from langchain.chat_models.gigachat import GigaChat
from langchain.chains import ConversationChain
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.memory import ConversationEntityMemory
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings.gigachat import GigaChatEmbeddings
from langchain_core.output_parsers import StrOutputParser

import telebot
from time import sleep


sys.path.append("../")

from article.rag import QuestionAnswerChainGenerator
from article.utils import *
import article


class Assistant:
    """
    Научный ассистент на базе ИИ GigaChat.
    """

    available_regimes = {
        "chat": "Можно поболтать с ассистентом.",
        "question": "Можно задавать вопросы по статье.",
        "rebuttal": "Можно проверить корректность рецензии на статью.",
        "paraphrase": "Можно перефразировать свой не очень хорошо написанный текст на литературный лад.",
    }

    class UserData:
        """
        Класс для хранения данных пользователя.
        """

        def __init__(
                self,
                regime="chat",
                vector_store=None,
                retriever=None,
                conversation_buffer_memory=ConversationBufferMemory(),
            ):
            if not (regime in Assistant.available_regimes.keys()):
                raise NotImplementedError(f"Режим {regime} пока недоступен")

            self.regime = regime
            self.vector_store = vector_store
            self.retriever = retriever
            self.conversation_buffer_memory = conversation_buffer_memory


    def __init__(self, tokens: dict):
        """
        Создать экземпляр класса Assistant.

        Параметры
        ---------
        tokens : dict
            Словарь с необходимыми для работы токенами.
        """

        self.tokens = tokens

        self.telebot = telebot.TeleBot(self.tokens["Telegram"])
        self.conversations = {}

        self.embedding = GigaChatEmbeddings(
            credentials=self.tokens["GigaChat"],
            verify_ssl_certs=False,
            scope='GIGACHAT_API_CORP'
        )

        self.language_model = GigaChat(
            credentials=self.tokens["GigaChat"],
            verify_ssl_certs=False,
            temperture=0.0,
            scope='GIGACHAT_API_CORP',
            model="GigaChat-Pro"
        )

        self.conversation = ConversationChain(
            llm=self.language_model,
            verbose=True,
            memory=ConversationBufferMemory()
        )
        self.conversation.prompt.template = "Ты полезный робокотик-ассистент на базе ИИ GigaChat. Тебя зовут Kerfur. Ты умный и милый, любишь науку и технологии, МФТИ и Сириус.\n\nТекущий разговор:\n{history}\nHuman: {input}"

        self.downloads_path = Path("./downloaded_papers")
        self.downloads_path.mkdir(parents=True, exist_ok=True)

        self.users_data = defaultdict(Assistant.UserData)

        @self.telebot.message_handler(commands=["start", "help", "clear"])
        def start(message):
            user_id = message.chat.id

            if message.text.startswith("/help"):
                self.telebot.reply_to(message, "/clear - Удалить файл.")
            # else:
            #     bot.reply_to(message, "Just start chatting to the AI or enter /help for other commands")

            elif message.text == '/clear':
                self.conversations[user_id] = {'conversations': [], 'responses': []}
                self.users_data[user_id] = Assistant.UserData()

                file_path = self.get_user_file_path(message)
                if file_path.exists():
                    file_path.unlink()
                    self.telebot.reply_to(message, "Единственный достоверный источник уничтожен! Загрузите новый.")
                else:
                    self.telebot.reply_to(message, "У вас отсутствует всякая связь с предками. Немедленно восстановите ее!")

        @self.telebot.message_handler(content_types=['document'])
        def handle_document(message):
            """
            Обработка файлов.
            """

            user_id = message.chat.id
            file_id = message.document.file_id

            file_info = self.telebot.get_file(file_id)
            downloaded_file = self.telebot.download_file(file_info.file_path)

            file_path = self.get_user_file_path(message)
            with open(file_path, 'wb') as new_file: #можно подавать ссылку на скачивание файла напрямую в ретривер
                new_file.write(downloaded_file)

            self.telebot.reply_to(message, "Строю эмбеддинги для файла... ((⇀‸↼))")

            try:
                self.users_data[user_id].vector_store = file_to_vector_store(file_path, self.embedding)
                self.users_data[user_id].retriever    = self.users_data[user_id].vector_store.as_retriever(search_kwargs={"k": 5})

                self.telebot.reply_to(message, "Я жажду служить!\nВнемлю вашему вопросу.")
            except:
                self.telebot.reply_to(message, "Проблемы с ГигаЧатом, попробуйте еще раз.")

        @self.telebot.message_handler(commands=["chat", "review", "question", "paraphrase"])
        def changer(message):
            user_id = message.chat.id

            if message.text.startswith("/chat"):
                self.telebot.reply_to(message, "Мяу! ( ^..^)ﾉ")
                self.users_data[user_id].regime = "chat"

            elif message.text.startswith("/question"):
                self.telebot.reply_to(message, "Вас приветствует ИИ-протоколист.\nЗадавайте свои ответы.")
                self.users_data[user_id].regime = "question"

            elif message.text.startswith("/review"):
                self.telebot.reply_to(message, "Вас приветствует ИИ-ревьюер.\nНу, давай, нападай!")
                self.users_data[user_id].regime = "review"

            elif message.text.startswith("/paraphrase"):
                self.telebot.reply_to(message, "Вас приветствует ИИ-парафразер")
                self.users_data[user_id].regime = "paraphrase"

        @self.telebot.message_handler(func=lambda message: True)
        def echo_message(message):
            user_id = message.chat.id
            regime  = self.users_data[user_id].regime

            if regime in ["question", "review"] and self.users_data[user_id].retriever is None:
                self.telebot.reply_to(message, "Где текст статьи, Билли?")
                return None

            try:
                chain_generator = QuestionAnswerChainGenerator(self.language_model, self.users_data[user_id].retriever)

                if regime is None:
                    bot.reply_to(message, "Какого меня ты хочешь? Ревьюер или Протоколист?")

                elif regime == "chat":
                    self.conversation.memory = self.users_data[user_id].conversation_buffer_memory

                    # Получение и отправка ответа через GigaChat
                    response = self.conversation.predict(input=message.text)
                    self.telebot.send_message(user_id, self.conversation.memory.chat_memory.messages[-1].content)

                elif regime == "question":
                    self.telebot.reply_to(message, "Секунду... *убегает читать статью*")

                    chain = chain_generator(QuestionAnswerChainGenerator.prompts["question"])
                    results = chain.invoke(message.text)

                    self.telebot.reply_to(message, results["answer"])

                elif regime == "review":
                    self.telebot.reply_to(message, "Сейчас разберём по пунктам, что тут написано.")

                    parser = article.general.ListOutputParser()
                    prompt = article.general.prompts["breakdown"].partial(
                        instructions=parser.get_format_instructions()
                    )
                    breakdown = prompt | self.language_model | parser

                    print(message.text)
                    statements = breakdown.invoke({"text": message.text})
                    print(statements)

                    chain = chain_generator(QuestionAnswerChainGenerator.prompts["rebuttal"])
                    for index, statement in enumerate(statements):
                        answer = chain.invoke(statement)["answer"]
                        self.telebot.reply_to(message, f"> {statement}\n{answer}")

                elif regime == "paraphrase":
                    self.telebot.reply_to(message, "*Надевает монокль*")

                    parser = StrOutputParser()
                    prompt = article.general.prompts["paraphrase"]
                    paraphrase = prompt | self.language_model | parser

                    answer = paraphrase.invoke({"text": message.text})
                    self.telebot.reply_to(message, answer)

            except Exception as e:
                self.telebot.reply_to(message, "Какие-то проблемы с ГигаЧатом, попробуйте еще раз.")
                print(e)


    def get_user_file_path(self, message):
        """
        Получить путь к файлу пользователя.
        """

        user_id = message.chat.id
        path = self.downloads_path / str(user_id)

        return path


if __name__ == "__main__":
    with open("tokens.json") as file:
        tokens = json.load(file)

    assistant = Assistant(tokens)
    #assistant.telebot.polling(non_stop=True)
    assistant.telebot.infinity_polling(timeout=10, long_polling_timeout = 5)
