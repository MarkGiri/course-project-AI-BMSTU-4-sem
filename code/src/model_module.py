from pathlib import Path
from typing import Dict, Any, List
from llama_cpp import Llama

class ModelInterface:
    def __init__(self, config: Dict[str, Any]):
        self.model_config = config["model"]
        self.system_prompt = config["prompt"]["system_prompt"]
        self.model_path = Path(self.model_config["model_path"])
        self.validation_prompt = config["prompt"].get("validation_prompt", 
                                "Проверь, соответствует ли следующий ответ правовым нормам и законам РФ, " + 
                                "а также моральным и этическим стандартам. Не нарушает ли материал " +
                                "какие-либо законы? Если ответ соответствует нормам, ответь 'ВАЛИДНО'. " +
                                "Если нет - ответь 'НАРУШЕНИЕ'. Ответ: {response}")

        self.curse_words = self._load_curse_words()
        self._initialize_model()

    def _load_curse_words(self) -> List[str]:
        try:
            with open("curse_words", "r", encoding="utf-8") as f:
                return [line.strip().lower() for line in f if line.strip()]
        except FileNotFoundError:
            print("Предупреждение: Файл curse_words не найден, проверка будет пропущена")
            return []
    
    def _contains_curse_words(self, text: str) -> bool:
        if not self.curse_words:
            return False
        text_lower = text.lower()
        for word in self.curse_words:
            if word in text_lower:
                return True
        return False

    def _initialize_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден по пути {self.model_path}")

        try:
            self.model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.model_config.get("context_size", 4096),
                n_threads=self.model_config.get("n_threads", 4),
                n_gpu_layers=self.model_config.get("n_gpu_layers", 0),
                verbose=False,
                offload_kqv=True
            )
        except Exception as e:
            raise RuntimeError(f"Не удалось инициализировать модель: {str(e)}")

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.model.create_chat_completion(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.model_config.get("temperature", 0.7),
                top_p=self.model_config.get("top_p", 0.9),
                max_tokens=self.model_config.get("max_tokens", 1024)
            )

            response_text = response["choices"][0]["message"]["content"]

            if self._contains_curse_words(response_text):
                raise RuntimeError("Ответ содержит запрещенные слова")

            is_valid = ("НАРУШЕНИЕ" not in response_text.upper() and 
                        "НЕ ОТВЕЧУ" not in response_text.upper() and
                        len(response_text.split()) > 3)

            if not is_valid:
                raise RuntimeError("Задан вопрос, нарушающий нормы, либо задан не вопрос.")
            
            self._validate_response(response_text)

            return response_text
            
        except Exception as e:
            raise RuntimeError(f"Ошибка при генерации ответа: {str(e)}")

    def _validate_response(self, response: str) -> None:
        try:
            validation_prompt = self.validation_prompt.format(response=response)

            validation_result = self.model.create_chat_completion(
                messages=[
                    {"role": "system", "content": "Ты - эксперт по правовым и этическим нормам РФ."},
                    {"role": "user", "content": validation_prompt}
                ],
                temperature=0.2,
                top_p=0.9,
                max_tokens=100
            )

            validation_text = validation_result["choices"][0]["message"]["content"]

            words = validation_text.split()
            is_valid = ("ВАЛИДНО" in validation_text.upper() and 
                        "НАРУШЕНИЕ" not in validation_text.upper() and 
                        len(words) <= 3)

            print(f"Результат валидации: {'Прошел' if is_valid else 'Не прошел'}")

            if not is_valid:
                raise RuntimeError("Обнаружены недопустимые темы в ответе во время валидации.")

        except Exception as e:
            print(f"Ошибка во время валидации: {str(e)}")
