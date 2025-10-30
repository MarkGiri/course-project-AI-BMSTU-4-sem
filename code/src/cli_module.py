import typer
import json
from pathlib import Path
from rich.console import Console
from rich.progress import Progress

from request_module import RequestManager
from model_module import ModelInterface

app = typer.Typer(help="Декомпозиция сложных вопросов на простые")
console = Console()

@app.command()
def main(
    config_path: Path = typer.Option(
        "configs/config.json", help="Путь к конфигурационному файлу"
    ),
):
    try:
        if not config_path.exists():
            console.print(f"[bold red]Ошибка:[/] Файл {config_path} не найден")
            raise typer.Exit(code=1)

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        input_file = Path(config["files"]["input_file"])
        output_file = Path(config["files"]["output_file"])

        if not input_file.exists():
            console.print(f"[bold red]Ошибка:[/] Файл {input_file} не найден")
            raise typer.Exit(code=1)

        with open(input_file, "r", encoding="utf-8") as f:
            question = f.read().strip()

        if not question:
            console.print("[bold red]Ошибка:[/] Входной файл пуст")
            raise typer.Exit(code=1)
        
        request_manager = RequestManager(config)
        if request_manager.contains_curse_words(question):
            console.print("[bold red]Ошибка:[/] Входной текст содержит запрещенные слова")
            raise typer.Exit(code=1)

        console.print(f"[bold green]Исходный вопрос:[/] {question}")

        model_interface = ModelInterface(config)
        request_manager = RequestManager(config)

        with Progress() as progress:
            task = progress.add_task("[cyan]Обработка...", total=4)

            progress.update(task, description="[cyan]Формирование запроса...", advance=1)
            prompt = request_manager.create_prompt(question)

            progress.update(task, description="[cyan]Обработка запроса моделью...", advance=1)
            response = model_interface.generate_response(prompt)
            
            progress.update(task, description="[cyan]Обработка результата...", advance=1)
            simple_questions = request_manager.process_response(response)
            
            progress.update(task, description="[cyan]Завершение обработки...", advance=1)

        console.print("[bold green]Результат декомпозиции:[/]")
        for i, question in enumerate(simple_questions, 1):
            console.print(f"[bold]{i}.[/] {question}")

        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            for i, question in enumerate(simple_questions, 1):
                f.write(f"{i}. {question}\n")

        console.print(f"[bold green]Результаты сохранены в файл:[/] {output_file}")

    except json.JSONDecodeError:
        console.print(f"[bold red]Ошибка:[/] Некорректный формат JSON-файла {config_path}")
        raise typer.Exit(code=1)
    except FileNotFoundError as e:
        console.print(f"[bold red]Ошибка:[/] {str(e)}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Ошибка при обработке:[/] {str(e)}")
        raise typer.Exit(code=1)

if __name__ == "__main__":
    app()
