"""Run CineMate in the terminal (or a Colab cell):   python app.py"""
from cinemate.agent import CineMate
from cinemate.data_loader import load


def main():
    print("Loading MovieLens data and training models...")
    bot = CineMate(load())
    print("\nCineMate: " + bot.greet())
    while True:
        try:
            text = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        answer = bot.reply(text)
        print("\nCineMate: " + answer)
        if text.strip().lower() in {"bye", "quit", "exit"}:
            break


if __name__ == "__main__":
    main()
