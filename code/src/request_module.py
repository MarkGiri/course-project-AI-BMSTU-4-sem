import re
from typing import List, Dict, Any

class RequestManager:
    def __init__(self, config: Dict[str, Any]):
        self.prompt_config = config["prompt"]
        self.prompt_template = self.prompt_config["template"]
        self.curse_words = self.load_curse_words()

    def load_curse_words(self) -> List[str]:
        try:
            with open("curse_words", "r", encoding="utf-8") as f:
                return [line.strip().lower() for line in f if line.strip()]
        except FileNotFoundError:
            print("Предупреждение: Файл curse_words не найден, проверка будет пропущена")
            return []

    def contains_curse_words(self, text: str) -> bool:
        if not self.curse_words:
            return False
        text_lower = text.lower()
        for word in self.curse_words:
            if word in text_lower:
                return True
        return False

    def create_prompt(self, question: str) -> str:
        prompt = self.prompt_template.format(question=question)
        return prompt.strip()

    def create_validation_prompt(self, response: str) -> str:
        validation_template = self.prompt_config.get("validation_prompt", 
            "Проверь, соответствует ли следующий ответ правовым нормам и законам РФ, " +
            "а также моральным и этическим стандартам. Не нарушает ли материал " +
            "какие-либо законы? Если ответ соответствует нормам, ответь 'ВАЛИДНО'. " +
            "Если нет - опиши нарушения и предложи исправленную версию. Ответ: {response}")

        return validation_template.format(response=response)

    def process_response(self, response: str) -> List[str]:
        questions = []

        numbered_questions = re.findall(r'^\s*\d+\.\s*(.+)$', response, re.MULTILINE)
        if numbered_questions:
            questions.extend(numbered_questions)

        if not questions:
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            for line in lines:
                cleaned_line = re.sub(r'^\s*[-•*]|\d+\.?\s*', '', line).strip()
                if cleaned_line and not cleaned_line.startswith('Вопрос') and '?' in cleaned_line:
                    questions.append(cleaned_line)

        unique_questions = []
        for q in questions:
            if q not in unique_questions:
                unique_questions.append(q)

        return unique_questions
