
import tkinter as tk
root = tk.Tk()
'''root.geometry("500x500")'''
root.resizable('0', '0')

canvas = tk.Canvas(root, bg='black', height=400, width=400)
canvas.grid(row=0, column=0, columnspan=4)

def draw(event):
    x , y = event.x , event.y
    x1 , y1 = x-20 , y-20
    x2, y2 = x+20 , y+20

    canvas.create_oval((x1, y1,x2,y2), fill='white', outline='white')

canvas.bind("<B1-Motion>", draw)

button_save = tk.Button(root, text='Save', width=15, height=2, bg='black', fg='white', font='Helvetica')
button_save.grid(row=2,column=0)

button_clear = tk.Button(root, text='Clear', width=15, height=2, bg='black', fg='white', font='Helvetica')
button_clear.grid(row=2,column=2)

button_exit= tk.Button(root, text='Exit', width=15, height=2, bg='black', fg='white', font='Helvetica')
button_exit.grid(row=2,column=3)

root.mainloop()