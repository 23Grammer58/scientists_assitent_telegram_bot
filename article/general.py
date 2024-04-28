import ast
from pathlib import Path
from operator import itemgetter

# Создание цепочек.
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

# Загрузка промптов.
from langchain.prompts import load_prompt
from langchain_core.prompts import ChatPromptTemplate

# Парсинг выходных данных.
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser


class ListOutputParser(BaseOutputParser):
    """
    Парсер выходных данных в Python list.
    """

    def parse(self, output: str) -> list:
        """
        Спарсить выходные данные.

        Параметры
        ---------
        output : str
            Строка выходных данных.

        Возвращает
        ----------
        output_as_list : list
            Список строк.
        """

        try:
            #output_as_list = output.strip('][').split(', ')
            output_as_list = ast.literal_eval(output)
        except:
            raise OutputParserException(f"Выражение не является Python list: {output}")

        return output_as_list

    def get_format_instructions(self) -> str:
        """
        Получить инструкции по форматированию.

        Возвращает
        ----------
        format_instructions : str
            Инструкции по форматированию.
        """

        #format_instructions = "Ответ приведи в виде списка, заключённого в квадратные скобки и разделённого запятыми. Например, [\"первый ответ\", \"второй ответ\", \"третий ответ\"]. Нельзя писать ничего, кроме этого списка. В качестве разделителя требуется использовать только запятую."
        format_instructions = "Каждый из ответов заключи в двойные кавычки и перечисли через запятую внутри квадратных скобок. Пример: [\"первый ответ\", \"второй ответ\", \"третий ответ\"]. Нельзя писать ничего, кроме этого списка."

        return format_instructions


prompts = {
    "breakdown"  : load_prompt(Path(__file__).parent / "prompts/breakdown.yaml"),
    "paraphrase" : load_prompt(Path(__file__).parent / "prompts/paraphrase.yaml"),
    "protocol" : load_prompt(Path(__file__).parent / "prompts/protocol.yaml"),
}

output_parsers = {
    "list" : ListOutputParser(),
}
