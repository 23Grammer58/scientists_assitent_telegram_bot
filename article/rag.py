from pathlib import Path
from operator import itemgetter

# Парсеры.
from langchain_core.output_parsers import StrOutputParser

# Создание цепочек.
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

# Загрузка промптов.
from langchain.prompts import load_prompt
from langchain_core.prompts import ChatPromptTemplate


class QuestionAnswerChainGenerator:
    """
    Класс, генерирующий цепочки вида вопрос-ответ на основе промпта.
    """

    prompts = {
        "question" : load_prompt(Path(__file__).parent / "prompts/question.yaml"),
        "rebuttal" : load_prompt(Path(__file__).parent / "prompts/rebuttal.yaml"),
        "protocol": load_prompt(Path(__file__).parent / "prompts/protocol.yaml"),
    }

    def __init__(self, language_model, retriever):
        """
        Создать экземпляр класса QuestionAnswerChainGenerator

        Параметры
        ---------
        language_model
            Языковая модель.
        retriever
            Извлекатель данных.
        """

        self.language_model = language_model
        self.retriever = retriever

    def format_snippets(self, documents: list) -> str:
        """
        Отформатировать фрагменты статьи в единый текст.

        Параметры
        ---------
        documents : list
            Список фрагментов.

        Возвращает
        ----------
        formated : str
            Итоговая строка, содержащая все фрагменты.
        """

        formatted = [
            f"Фрагмент статьи: {document.page_content}"
            for document in documents
        ]

        formatted = "\n\n" + "\n\n".join(formatted)

        return formatted

    def __call__(self, prompt=None):
        """
        Создаёт цепочку для ответов на вопросы по статье.

        Параметры
        ---------
        prompt : , optional
            Промпт, для которого требуется создать цепочку.
            По умолчанию соответствует цепочке ответов на вопросы.
        """

        # По умолчанию промпт соответствует ответам на вопросы о статье.
        if prompt is None:
            prompt = QuestionAnswerChainGenerator.prompts["question_answer"]

        # На вход подаётся retriever, с вырезками из статьи.
        format_input = itemgetter("documents") | RunnableLambda(self.format_snippets)

        # Для ответа вырезки подставляются в промпт и подаются на вход языковой модели.
        answer = prompt | self.language_model | StrOutputParser()

        chain = (
            RunnableParallel(question=RunnablePassthrough(), documents=self.retriever)
            .assign(context=format_input)
            .assign(answer=answer)
            .pick(["answer", "documents"])
        )

        return chain

