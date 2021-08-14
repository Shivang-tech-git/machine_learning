import PyPDF2
from tkinter import messagebox
from tkinter import filedialog
import tkinter as tk
import io

def pdf_reader():
    # creating a pdf file object
    pdfFileObj = open(filename, 'rb')
    # creating a pdf reader object
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    with io.open("Extract.txt", "w", encoding="utf-8") as f:
        for page_count in range(0,pdfReader.numPages):
            # creating a page object
            pageObj = pdfReader.getPage(page_count)
            # extracting text from page
            f.write(pageObj.extractText())
    pdfFileObj.close()
    messagebox.showinfo('PDF TO TEXT TOOL',
                        'Please see extract.txt file for output.')

def pdf_file():
    global filename
    filename = filedialog.askopenfilename(initialdir="/",
                                          title="Select PDF File")

if __name__ == '__main__':
    window = tk.Tk()
    window.title('AXA XL')
    # ------------------------------- Title ----------------------------
    greeting = tk.Label(text="PDF TO TEXT TOOL", font=('Arial',12), fg='yellow', bg='green', width=60)
    frm_open = tk.Frame(master=window, width=60)
    # ------------------------------ Browse allocation file --------------------------
    lbl_open = tk.Label(master=frm_open, text='Select PDF file',font=('Arial',10),
                        width=30, borderwidth=2, relief="ridge").grid(row=0, column=0, pady=5)
    btn_open = tk.Button(master=frm_open, text='Browse',font=('Arial',10), width=15, bg='lavender',
                         command=pdf_file).grid(row=0,column=1,pady=5)
    btn_extract = tk.Button(master=frm_open, text='Run', font=('Arial', 10), width=15, bg='lavender',
                         command=pdf_reader).grid(row=0, column=2, pady=5)

    greeting.pack()
    frm_open.pack(fill=tk.BOTH)
    # lbl_status.pack(pady=5, side=tk.BOTTOM)
    window.mainloop()