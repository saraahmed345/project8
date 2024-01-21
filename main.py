from sklearn import preprocessing
import tkinter as tk
from tkinter import ttk
import numpy as np
import Back_propagation as bp
import BackPropagation_tanh as bt

def home():
    home=tk.Tk()
    home.geometry("600x500")
    home.title("Neural netwark task 2")
    l1 = tk.Label(home, text=" Enter number of hidden layers")
    l1.pack(pady=5)
    hl = tk.Text(home, width=20, height=1)
    hl.pack(padx=5)
    l2 = tk.Label(home, text=" Enter number of neurons in each hidden layer")
    l2.pack(pady=5)
    neuron = tk.Text(home, width=20, height=1)
    neuron.pack(padx=5)
    l4= tk.Label(home, text="Enter learning rate(eta)")
    l4.pack(pady=5)
    eta=tk.Text(home,width=20,height=1)
    eta.pack(padx=5)

    l5 = tk.Label(home, text="Enter number of epochs (m)")
    l5.pack(pady=5)
    m = tk.Text(home, width=20, height=1)
    m.pack(padx=5)
    # Set the default selection
    Checkbutton1 = tk.IntVar()
    button1 = tk.Checkbutton(home, text="bais or not",
                         variable=Checkbutton1,
                         onvalue=1,
                         offvalue=0,
                         height=2,
                         width=10)
    button1.pack()
    v = tk.StringVar(home, "1")

    # Dictionary to create multiple buttons
    values = {"Sigmoid ": "1",
              "Hyperbolic Tangent sigmoid ": "2"}
    # Loop is used to create multiple Radiobuttons
    # rather than creating each button separately
    for (text, value) in values.items():
        tk.Radiobutton(home, text=text, variable=v,
                    value=value).pack(pady=5)
    def dotask():
        function = v.get()
        layers=int(hl.get("1.0", "end-1c"))
        neurons=int(neuron.get("1.0", "end-1c"))
        et=float(eta.get("1.0", "end-1c"))
        epochs=int(m.get("1.0", "end-1c"))
        bais=Checkbutton1.get()
        if function=="1":
            bp.sig(layers,et,epochs)
        elif function=="2":
            bt.ta(layers,epochs,et)
    btn=tk.Button(home,text="Do task",command=dotask)
    btn.pack()
    home.mainloop()
home()