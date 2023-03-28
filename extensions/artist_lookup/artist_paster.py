import os
import sys
from typing import Optional
import pyperclip
import tkinter as tk

"""
TASK: Copy the contents of a text file to the clipboard one line at a time.
USAGE: Run the script and click the "Paste" button to copy the next line to the clipboard.
"""
class LineCopier(tk.Tk):
    def __init__(self, file_path: str):
        super().__init__()

        self.file_path = file_path
        self.lines = self._read_lines_from_file()
        self.current_line = 0

        self.title("Line Copier")
        self.geometry("200x100")
        self.resizable(False, False)

        self.paste_button = tk.Button(self, text="Paste", command=self.copy_next_line)
        self.paste_button.pack(expand=True, pady=20)

    def _read_lines_from_file(self) -> Optional[list]:
        if not os.path.isfile(self.file_path):
            print(f"The file '{self.file_path}' does not exist.")
            sys.exit(1)

        with open(self.file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        return lines

    def copy_next_line(self):
        if self.current_line < len(self.lines):
            pyperclip.copy(self.lines[self.current_line].strip())
            print(f"Copied line {self.current_line + 1} to clipboard.")
            self.current_line += 1
        else:
            print("Reached the end of the file.")


if __name__ == "__main__":
    file_path = "D:\Andrew\Pictures\\artists_v1.txt"
    line_copier = LineCopier(file_path)
    line_copier.mainloop()