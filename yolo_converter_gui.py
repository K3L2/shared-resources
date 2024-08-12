import tkinter as tk
from tkinter import filedialog, messagebox
from yolo_converter import Labelme2YOLO

class Labelme2YOLOGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Labelme to YOLO Converter")

        self.label_txt_path = None

        tk.Label(root, text="JSON Directory:").grid(row=0, column=0)
        self.json_dir_entry = tk.Entry(root, width=40)
        self.json_dir_entry.grid(row=0, column=1)
        tk.Button(root, text="Browse", command=self.browse_json_dir).grid(row=0, column=2)

        tk.Label(root, text="Validation Set Size (0.0 - 1.0):").grid(row=1, column=0)
        self.val_size_entry = tk.Entry(root, width=10)
        self.val_size_entry.grid(row=1, column=1)
        self.val_size_entry.insert(0, "0.1")

        self.segmentation_var = tk.BooleanVar()
        tk.Checkbutton(root, text="Convert to Segmentation", variable=self.segmentation_var).grid(row=2, columnspan=2)

        tk.Label(root, text="Label Mapping TXT:").grid(row=3, column=0)
        self.label_txt_entry = tk.Entry(root, width=40)
        self.label_txt_entry.grid(row=3, column=1)
        tk.Button(root, text="Load", command=self.load_label_txt).grid(row=3, column=2)

        tk.Label(root, text="Output Folder Name:").grid(row=4, column=0)
        self.folder_name_entry = tk.Entry(root, width=40)
        self.folder_name_entry.grid(row=4, column=1)
        self.folder_name_entry.insert(0, "YOLODataset")

        tk.Button(root, text="Convert", command=self.convert).grid(row=5, columnspan=3)

    def browse_json_dir(self):
        self.json_dir_entry.delete(0, tk.END)
        self.json_dir_entry.insert(0, filedialog.askdirectory())

    def load_label_txt(self):
        self.label_txt_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        self.label_txt_entry.delete(0, tk.END)
        self.label_txt_entry.insert(0, self.label_txt_path)
        if self.label_txt_path:
            with open(self.label_txt_path, 'r') as file:
                labels = file.read().strip().splitlines()
                messagebox.showinfo("Label Mapping Loaded", f"Labels:\n{', '.join(labels)}")

    def convert(self):
        json_dir = self.json_dir_entry.get()
        val_size = float(self.val_size_entry.get())
        to_seg = self.segmentation_var.get()
        folder_name = self.folder_name_entry.get()

        if not json_dir:
            messagebox.showerror("Error", "Please select a JSON directory.")
            return

        convertor = Labelme2YOLO(json_dir, label_txt_path=self.label_txt_path, to_seg=to_seg, folder_name=folder_name)
        convertor.convert(val_size)
        messagebox.showinfo("Success", "Conversion completed successfully!")


if __name__ == "__main__":
    root = tk.Tk()
    app = Labelme2YOLOGUI(root)
    root.mainloop()
