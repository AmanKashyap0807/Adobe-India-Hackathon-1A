import tkinter as tk
from tkinter import ttk

# Absolute minimum code to test tkinter functionality
def main():
    root = tk.Tk()
    root.geometry("800x600")
    root.title("Basic Tkinter Test")
    
    label = tk.Label(root, text="If you can see this, tkinter is working", font=("Arial", 16))
    label.pack(pady=50)
    
    button = ttk.Button(root, text="Click Me", command=lambda: label.config(text="Button clicked!"))
    button.pack(pady=20)
    
    root.mainloop()

if __name__ == "__main__":
    main()
