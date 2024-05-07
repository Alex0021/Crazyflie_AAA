import tkinter as tk
from PIL import Image, ImageTk
from tkinter import ttk

class App(tk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.mainframe = root
        self.img = Image.open("bitcraze_cf2.png")
        self.mainframe.columnconfigure((0,1), weight=1)
        self.mainframe.rowconfigure(0, weight=1)
        self.mainframe.rowconfigure((1,2), weight=4)
        self._init_components()

    
    def _init_components(self):
        self.label_title = tk.Label(self.mainframe, text="Crazyflie Controller", font=("Arial", 24), bg="darkgrey")
        self.label_title.grid(row=0, column=0, columnspan=2, sticky="news")



if __name__ == '__main__':
    root = tk.Tk()
    root.geometry('800x600')
    root.title("Crazyflie Controller")
    app = App(root)
    root.mainloop()