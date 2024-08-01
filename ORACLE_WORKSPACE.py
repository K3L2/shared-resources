import tkinter as tk
from tkinter import messagebox, filedialog
import cx_Oracle
import pandas as pd
import json


# DB 연결 함수
def connect_to_db(ip, port, sid, username, password):
    try:
        dsn_tns = cx_Oracle.makedsn(ip, port, sid)
        conn = cx_Oracle.connect(username, password, dsn_tns)
        return conn
    except cx_Oracle.DatabaseError as e:
        error, = e.args
        messagebox.showerror("Database Error", f"Error Code: {error.code}\nError Message: {error.message}")
        return None


# 쿼리 실행 함수
def execute_query():
    ip = ip_entry.get()
    port = port_entry.get()
    sid = sid_entry.get()
    username = username_entry.get()
    password = password_entry.get()
    query = query_text.get("1.0", tk.END)

    if not (ip and port and sid and username and password and query.strip()):
        messagebox.showwarning("Input Error", "Please fill all fields.")
        return

    conn = connect_to_db(ip, port, sid, username, password)
    if conn:
        cursor = conn.cursor()
        try:
            cursor.execute(query)
            if query.strip().upper().startswith("SELECT"):
                columns = [col[0] for col in cursor.description]
                data = cursor.fetchall()
                df = pd.DataFrame(data, columns=columns)
                result_text.delete("1.0", tk.END)
                result_text.insert(tk.END, df.to_string())
            else:
                conn.commit()
                result_text.delete("1.0", tk.END)
                result_text.insert(tk.END, "Query executed successfully.")
        except cx_Oracle.DatabaseError as e:
            error, = e.args
            messagebox.showerror("Database Error", f"Error Code: {error.code}\nError Message: {error.message}")
        finally:
            cursor.close()
            conn.close()


# 설정 저장 함수
def save_config():
    config = {
        "ip": ip_entry.get(),
        "port": port_entry.get(),
        "sid": sid_entry.get(),
        "username": username_entry.get(),
        "password": password_entry.get()
    }
    save_path = filedialog.asksaveasfilename(defaultextension=".json",
                                             filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
    if save_path:
        with open(save_path, 'w') as file:
            json.dump(config, file)
        messagebox.showinfo("Save Configuration", "Configuration saved successfully.")


# 설정 불러오기 함수
def load_config():
    load_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
    if load_path:
        with open(load_path, 'r') as file:
            config = json.load(file)
        ip_entry.delete(0, tk.END)
        ip_entry.insert(0, config.get("ip", ""))
        port_entry.delete(0, tk.END)
        port_entry.insert(0, config.get("port", ""))
        sid_entry.delete(0, tk.END)
        sid_entry.insert(0, config.get("sid", ""))
        username_entry.delete(0, tk.END)
        username_entry.insert(0, config.get("username", ""))
        password_entry.delete(0, tk.END)
        password_entry.insert(0, config.get("password", ""))
        messagebox.showinfo("Load Configuration", "Configuration loaded successfully.")


# GUI 설정
root = tk.Tk()
root.title("Oracle DB Query Executor")

tk.Label(root, text="IP Address").grid(row=0, column=0, pady=3)
tk.Label(root, text="Port").grid(row=1, column=0, pady=3)
tk.Label(root, text="SID").grid(row=2, column=0, pady=3)
tk.Label(root, text="Username").grid(row=3, column=0, pady=3)
tk.Label(root, text="Password").grid(row=4, column=0, pady=3)

ip_entry = tk.Entry(root)
port_entry = tk.Entry(root)
sid_entry = tk.Entry(root)
username_entry = tk.Entry(root)
password_entry = tk.Entry(root, show='*')

ip_entry.grid(row=0, column=1, pady=3)
port_entry.grid(row=1, column=1, pady=3)
sid_entry.grid(row=2, column=1, pady=3)
username_entry.grid(row=3, column=1, pady=3)
password_entry.grid(row=4, column=1, pady=3)

save_button = tk.Button(root, text="Save Config", command=save_config)
load_button = tk.Button(root, text="Load Config", command=load_config)
save_button.grid(row=5, column=0, pady=3)
load_button.grid(row=5, column=1, pady=3)

tk.Label(root, text="Query").grid(row=6, column=0, pady=3)
query_text = tk.Text(root, height=10, width=50)
query_text.grid(row=7, columnspan=2, pady=3)

execute_button = tk.Button(root, text="Execute", command=execute_query)
execute_button.grid(row=8, columnspan=2, pady=3)

tk.Label(root, text="Result").grid(row=9, column=0, pady=3)
result_text = tk.Text(root, height=15, width=80)
result_text.grid(row=10, columnspan=3, pady=3)

scroll_y = tk.Scrollbar(root, orient='vertical', command=result_text.yview)
scroll_y.grid(row=10, column=3, sticky='ns')
result_text.config(yscrollcommand=scroll_y.set)

root.mainloop()
