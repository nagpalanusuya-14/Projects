import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json, os, shutil, zipfile, re
from datetime import datetime, timedelta
import random
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------------- files & constants ----------------
USERS_FILE = "users.json"
ABOUT_FILE = "about_app.txt"
REPORTS_DIR = "weekly_reports"
EXPORT_DIR = "exports"
DANGER_THRESHOLD = 9.0

# ---------------- utilities ----------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_json_safe(path, default):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, indent=2)
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # backup corrupt then recreate
        backup = path + ".corrupt_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.move(path, backup)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, indent=2)
        return default

def save_json_safe(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def sanitize_filename(s):
    return "".join(c for c in s if c.isalnum() or c in (" ", "_", "-")).rstrip().replace(" ", "_")

# ---------------- users ----------------
def load_users():
    default = {"admin": {"password": "admin123", "role": "admin"}}
    data = load_json_safe(USERS_FILE, default)
    # migrate legacy simple format
    migrated = False
    if isinstance(data, dict):
        for k, v in list(data.items()):
            if isinstance(v, str):
                migrated = True
                break
    if migrated:
        new = {}
        for k, v in data.items():
            if isinstance(v, str):
                new[k] = {"password": v, "role": "user"}
            elif isinstance(v, dict):
                pw = v.get("password") or v.get("pw") or ""
                role = v.get("role", "user")
                new[k] = {"password": pw, "role": role}
            else:
                new[k] = {"password": "", "role": "user"}
        save_json_safe(USERS_FILE, new)
        return new
    return data

def save_users(users):
    save_json_safe(USERS_FILE, users)

# ---------------- dataset ----------------
def make_dates(start, days):
    return [start + timedelta(days=i) for i in range(days)]

def load_groundwater_df():
    dates = make_dates(datetime(2025, 10, 1), 10)

    states = (['Rajasthan'] * 10 +
              ['Gujarat'] * 10 +
              ['Punjab'] * 10 +
              ['Tamil Nadu'] * 10 +
              ['Meghalaya'] * 10)

    sites = (['Jaipur - Well A'] * 10 +
             ['Ahmedabad - Well B'] * 10 +
             ['Ludhiana - Well C'] * 10 +
             ['Coimbatore - Well D'] * 10 +
             ['Shillong - Well E'] * 10)

    levels = [
        14.2, 14.18, 14.15, 14.10, 14.05, 13.98, 13.90, 13.80, 13.68, 13.55,
        11.8, 11.79, 11.75, 11.73, 11.70, 11.67, 11.63, 11.58, 11.50, 11.42,
        8.4, 8.38, 8.36, 8.34, 8.31, 8.28, 8.24, 8.20, 8.15, 8.10,
        16.5, 16.48, 16.46, 16.43, 16.40, 16.36, 16.32, 16.25, 16.18, 16.10,
        9.2, 9.18, 9.17, 9.15, 9.13, 9.10, 9.08, 9.05, 9.02, 8.98
    ]

    df = pd.DataFrame({
        "timestamp": dates * 5,
        "state": states,
        "location": sites,
        "water_level_m": levels
    })
    return df.sort_values(["state", "timestamp"]).reset_index(drop=True)

# ---------------- risk & recommendation ----------------
def risk_category(level):
    if level >= 15:
        return "LOW", "#10B981"
    if 12 <= level < 15:
        return "MEDIUM", "#F59E0B"
    if 9 <= level < 12:
        return "HIGH", "#EF4444"
    return "CRITICAL", "#7F1D1D"

def ai_recommendation_for_state(current_level):
    label, _ = risk_category(current_level)
    recs = []
    if label == "LOW":
        recs.append("Maintain current management; monitor monthly.")
        recs.append("Consider recharge initiatives proactively.")
    elif label == "MEDIUM":
        recs.append("Encourage drip/scheduled irrigation.")
        recs.append("Construct recharge pits locally.")
    elif label == "HIGH":
        recs.append("Restrict high-water-demand crops.")
        recs.append("Start artificial recharge programs.")
    else:
        recs.append("Declare emergency conservation measures.")
        recs.append("Deploy temporary water supply and fast recharge.")

# ---------------- forecasting helpers ----------------
def linear_fit_stats(x, y):
    # x: ordinal ints, y: values
    p = np.polyfit(x, y, 1)
    yhat = np.polyval(p, x)
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2) if y.size>0 else 1.0
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    return float(p[0]), float(p[1]), float(np.clip(r2, 0.0, 1.0))

def forecast_enhanced(df_state, days=30):
    df_state = df_state.sort_values("timestamp")
    if df_state.shape[0] < 3:
        return [], [], 0.0
    x = np.array([d.toordinal() for d in df_state["timestamp"]])
    y = np.array(df_state["water_level_m"], dtype=float)
    slope, intercept, r2 = linear_fit_stats(x, y)
    last_date = df_state["timestamp"].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, days+1)]
    xf = np.array([d.toordinal() for d in future_dates])
    yf = np.polyval([slope, intercept], xf)
    # don't allow negative
    yf = [float(max(0.0, float(v))) for v in yf]
    return future_dates, yf, r2

def ai_alerts_for_state(df_state, forecast_days=30, danger_threshold=DANGER_THRESHOLD):
    dates, preds, conf = forecast_enhanced(df_state, days=forecast_days)
    alerts = [(d, v) for d, v in zip(dates, preds) if v < danger_threshold]
    return alerts, conf

# ---------------- simple NLP helpers ----------------
_WORD_TO_NUM = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,"thirty":30,"sixty":60,"hundred":100}
def words_to_number(text):
    text = text.lower()
    total = 0; last=0
    for tok in re.findall(r'\w+', text):
        if tok in _WORD_TO_NUM:
            val = _WORD_TO_NUM[tok]
            if val == 100:
                last = last*100 if last else 100
            else:
                last += val
        else:
            if last:
                total += last; last=0
    total += last
    return total if total>0 else None

def extract_days_from_text(m):
    n = words_to_number(m)
    if n: return int(n)
    nums = re.findall(r'\d+', m)
    if nums: return int(nums[0])
    if "month" in m: return 30
    if "year" in m: return 365
    return 30

def fuzzy_find_state(m, df):
    ml = m.lower()
    states = df["state"].unique().tolist()
    for s in states:
        if s.lower() in ml:
            return s
    # word-wise
    for s in states:
        for token in s.lower().split():
            if token in ml:
                return s
    return None

def ai_chat_answer_smart(message, df):
    m = (message or "").strip().lower()
    if not m:
        return "Please enter a query, e.g., 'predict Meghalaya 90'."
    if any(g in m for g in ("hello","hi","hey")):
        return "Hi! Try: predict <state> <days>, alerts <days>, recommend <state>, summary"
    if "summary" in m:
        states = sorted(df["state"].unique())
        latest = df.sort_values("timestamp").groupby("state").tail(1)
        out = f"Rows: {len(df)}. States: {', '.join(states)}.\nLatest:\n"
        for _, r in latest.iterrows():
            out += f"- {r['state']}: {r['water_level_m']:.2f} m\n"
        return out
    if "predict" in m or "forecast" in m:
        days = extract_days_from_text(m)
        state = fuzzy_find_state(m, df)
        if not state:
            return "Please mention a state to predict."
        sub = df[df["state"]==state].sort_values("timestamp")
        if sub.empty:
            return f"No data for {state}."
        dates, preds, conf = forecast_enhanced(sub, days=days)
        if not dates:
            return "Not enough history to forecast."
        out = f"Forecast for {state} (next {days} days) — confidence {conf*100:.0f}%:\n"
        if days > 60:
            for d, v in list(zip(dates, preds))[:5]:
                out += f"- {d.strftime('%Y-%m-%d')}: {v:.2f} m\n"
            out += "...\n"
            for d,v in list(zip(dates, preds))[-5:]:
                out += f"- {d.strftime('%Y-%m-%d')}: {v:.2f} m\n"
        else:
            for d, v in zip(dates, preds):
                out += f"- {d.strftime('%Y-%m-%d')}: {v:.2f} m\n"
        if preds[-1] < DANGER_THRESHOLD:
            out += f"\n⚠ ALERT: Projected level drops below {DANGER_THRESHOLD} m."
        return out
    if "alert" in m:
        days = extract_days_from_text(m)
        msg = ""
        for s in sorted(df["state"].unique()):
            sub = df[df["state"]==s].sort_values("timestamp")
            alerts, conf = ai_alerts_for_state(sub, forecast_days=days)
            if alerts:
                msg += f"{s} alerts:\n"
                for d, v in alerts[:5]:
                    msg += f"- {d.strftime('%Y-%m-%d')}: {v:.2f} m\n"
        return msg or f"No alerts predicted in next {days} days."
    if "recommend" in m:
        state = fuzzy_find_state(m, df)
        if not state:
            return "Please mention a state e.g., 'recommend Gujarat'."
        sub = df[df["state"]==state].sort_values("timestamp")
        cur = float(sub["water_level_m"].iloc[-1])
        recs = ai_recommendation_for_state(cur)
        return f"Recommendations for {state} (level {cur:.2f} m):\n- " + "\n- ".join(recs or [])
    return "I didn't understand. Try 'help' or 'predict <state> <days>'."
class GroundwaterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GW App — Groundwater Flagship (A)")
        self.geometry("1280x820")
        self.configure(bg="#0b1220")

        # load data & users
        self.users = load_users()
        self.df = load_groundwater_df()
        self.current_user = None

        # layout: sidebar (hidden until login) and container for pages
        self.sidebar = tk.Frame(self, bg="#0f1724", width=220)
        # don't place until login
        self.sidebar.place_forget()

        self.container = tk.Frame(self, bg="#0b1220")
        self.container.place(relx=0.0, rely=0.0, relwidth=1.0, relheight=1.0)

        # page registry
        self.pages = {}

        # create and show login page immediately
        self.login_page = LoginPage(self.container, self)
        self.login_page.place(relx=0, rely=0, relwidth=1, relheight=1)

    def register_pages(self):
        """
        Instantiate pages after login (safe to call once).
        Creates pages only if their class is defined.
        """
        page_classes = {
            "DashboardPage": DashboardPage,
            "TrendsPage": TrendsPage,
            "SimulatorPage": SimulatorPage,
            "AICenterPage": AICenterPage,
            "TablePage": TablePage,
            "ReportsPage": ReportsPage,
            "AboutPage": AboutPage,
            "SettingsPage": SettingsPage,
            "RechargeCalculatorPage": RechargeCalculatorPage,
            "SmartAlertsPage": SmartAlertsPage
        }
        for name, cls in page_classes.items():
            if name in self.pages:
                continue
            try:
                page = cls(self.container, self)
                page.place(relx=0, rely=0, relwidth=1, relheight=1)
                self.pages[name] = page
            except Exception as e:
                print(f"[WARN] Could not initialize page {name}: {e}")

        # after registration, show dashboard if present
        if "DashboardPage" in self.pages:
            self.pages["DashboardPage"].tkraise()

    def show_sidebar(self):
        if not self.sidebar.winfo_ismapped():
            self.sidebar.place(relx=0.0, rely=0.0, relwidth=0.17, relheight=1.0)
            self._build_sidebar()
            # move container to the right
            self.container.place(relx=0.17, rely=0.0, relwidth=0.83, relheight=1.0)

    def hide_sidebar(self):
        if self.sidebar.winfo_ismapped():
            self.sidebar.place_forget()
        self.container.place(relx=0.0, rely=0.0, relwidth=1.0, relheight=1.0)

    def _build_sidebar(self):
        for w in self.sidebar.winfo_children():
            w.destroy()
        tk.Label(self.sidebar, text="GW Dashboard", bg="#0f1724", fg="white", font=("Segoe UI", 14, "bold")).pack(pady=12, anchor="w", padx=10)
        def add_btn(text, page_key):
            b = tk.Button(self.sidebar, text="  "+text, anchor="w", bd=0, bg="#0f1724", fg="white", padx=12,
                          command=lambda k=page_key: self._nav(k))
            b.pack(fill="x", pady=4, padx=8)
            b.bind("<Enter>", lambda e, btn=b: btn.config(bg="#143b57"))
            b.bind("<Leave>", lambda e, btn=b: btn.config(bg="#0f1724"))

        add_btn("Overview", "DashboardPage")
        add_btn("Trends", "TrendsPage")
        add_btn("Simulator", "SimulatorPage")
        add_btn("AI Center", "AICenterPage")
        add_btn("Data Table", "TablePage")
        add_btn("Weekly Reports", "ReportsPage")
        add_btn("Smart Alerts", "SmartAlertsPage")
        add_btn("Recharge Calculator", "RechargeCalculatorPage")
        add_btn("About", "AboutPage")
        add_btn("Settings", "SettingsPage")

        # logout
        tk.Frame(self.sidebar, height=20, bg="#0f1724").pack(fill="x", pady=8)
        def _logout():
            if messagebox.askyesno("Logout", "Return to login?"):
                self.current_user = None
                self.hide_sidebar()
                self.login_page.reset_and_show()
        tk.Button(self.sidebar, text="  Logout", anchor="w", bd=0, bg="#0f1724", fg="white", padx=12, command=_logout).pack(fill="x", pady=6, padx=8)

    def _nav(self, page_key):
        if not self.current_user:
            messagebox.showwarning("Login required", "Please login to access pages.")
            self.login_page.tkraise()
            return
        page = self.pages.get(page_key)
        if page:
            page.tkraise()
        else:
            messagebox.showwarning("Unavailable", f"Page '{page_key}' not available. Ensure all parts loaded.")


# ---------------- Login, Create, Forgot pages ----------------
class LoginPage(tk.Frame):
    def __init__(self, parent, app: GroundwaterApp):
        super().__init__(parent, bg="#0b1220")
        self.app = app
        self._build()

    def _build(self):
        card = tk.Frame(self, bg="#0f1724", padx=20, pady=20)
        card.place(relx=0.5, rely=0.5, anchor="center", width=620, height=420)
        tk.Label(card, text="GW App — Login", bg="#0f1724", fg="white", font=("Segoe UI", 18, "bold")).pack(pady=(0,8))
        frm = tk.Frame(card, bg="#0f1724"); frm.pack()
        tk.Label(frm, text="Username", bg="#0f1724", fg="#e6eef6").grid(row=0, column=0, sticky="w", pady=6)
        self.user_entry = ttk.Entry(frm, width=36); self.user_entry.grid(row=0, column=1, padx=8, pady=6); self.user_entry.insert(0, "admin")
        tk.Label(frm, text="Password", bg="#0f1724", fg="#e6eef6").grid(row=1, column=0, sticky="w", pady=6)
        self.pw_entry = ttk.Entry(frm, show="•", width=36); self.pw_entry.grid(row=1, column=1, padx=8, pady=6); self.pw_entry.insert(0, "admin123")
        btnf = tk.Frame(card, bg="#0f1724"); btnf.pack(pady=12)
        ttk.Button(btnf, text="Login", command=self._login).pack(side="left", padx=6)
        ttk.Button(btnf, text="Create Account", command=self._open_create).pack(side="left", padx=6)
        ttk.Button(btnf, text="Forgot Password", command=self._open_forgot).pack(side="left", padx=6)
        ttk.Label(card, text="(Default: admin / admin123)", foreground="#94a3b8", background="#0f1724").pack(pady=(8,0))

    def _login(self):
        u = self.user_entry.get().strip(); p = self.pw_entry.get().strip()
        users = load_users()
        if u in users and users[u].get("password") == p:
            self.app.current_user = u
            self.place_forget()
            self.app.show_sidebar()
            self.app.register_pages()
            # show dashboard if present
            if "DashboardPage" in self.app.pages:
                self.app.pages["DashboardPage"].tkraise()
            else:
                messagebox.showinfo("Welcome", f"Welcome {u}")
        else:
            messagebox.showerror("Login failed", "Incorrect username or password.")

    def _open_create(self):
        # if CreateAccountPage is registered show it; otherwise pop up a small dialog
        if "CreateAccountPage" in self.app.pages:
            self.app.pages["CreateAccountPage"].tkraise()
            return
        dlg = CreateAccountDialog(self)
        dlg.grab_set()

    def _open_forgot(self):
        if "ForgotPasswordPage" in self.app.pages:
            self.app.pages["ForgotPasswordPage"].tkraise()
            return
        dlg = ForgotPasswordDialog(self)
        dlg.grab_set()

    def reset_and_show(self):
        self.user_entry.delete(0, "end"); self.pw_entry.delete(0, "end")
        self.user_entry.insert(0, "admin"); self.pw_entry.insert(0, "admin123")
        self.app.current_user = None
        self.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.tkraise()

class CreateAccountDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Create Account")
        self.geometry("360x240")
        ttk.Label(self, text="Username").pack(pady=(12,0))
        self.u = ttk.Entry(self); self.u.pack(pady=6)
        ttk.Label(self, text="Password").pack(pady=(8,0))
        self.p = ttk.Entry(self, show="•"); self.p.pack(pady=6)
        ttk.Label(self, text="Role (admin/user)").pack(pady=(8,0)); self.r = ttk.Entry(self); self.r.pack(pady=6); self.r.insert(0,"user")
        ttk.Button(self, text="Create", command=self.create).pack(pady=12)

    def create(self):
        u = self.u.get().strip(); p = self.p.get().strip(); r = self.r.get().strip() or "user"
        if not u or not p:
            messagebox.showwarning("Missing", "Enter username and password.")
            return
        users = load_users()
        if u in users:
            messagebox.showerror("Exists", "Username already exists.")
            return
        users[u] = {"password": p, "role": r}
        save_users(users)
        messagebox.showinfo("Created", "Account created.")
        self.destroy()

class ForgotPasswordDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Reset Password")
        self.geometry("360x200")
        ttk.Label(self, text="Username").pack(pady=(12,0)); self.u = ttk.Entry(self); self.u.pack(pady=6)
        ttk.Label(self, text="New Password").pack(pady=(8,0)); self.p = ttk.Entry(self, show="•"); self.p.pack(pady=6)
        ttk.Button(self, text="Reset", command=self.reset).pack(pady=12)
    def reset(self):
        u = self.u.get().strip(); p = self.p.get().strip()
        users = load_users()
        if u not in users:
            messagebox.showerror("Not found", "Username not found.")
            return
        users[u]["password"] = p
        save_users(users)
        messagebox.showinfo("Done", "Password reset.")
        self.destroy()
# Dashboard
class DashboardPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg="white")
        self.app = app
        self._build()

    def _build(self):
        tk.Label(self, text="Overview — 10-day Depletion Summary", font=("Segoe UI", 14, "bold"), bg="white").pack(anchor="w", padx=12, pady=8)
        card_frame = tk.Frame(self, bg="white"); card_frame.pack(fill="x", padx=12)
        for s in sorted(self.app.df["state"].unique()):
            f = tk.Frame(card_frame, bg="#eef2ff", relief="raised", bd=1)
            f.pack(side="left", expand=True, fill="x", padx=6, pady=6)
            block = self.app.df[self.app.df["state"]==s].sort_values("timestamp")
            dep = float(block["water_level_m"].iloc[0] - block["water_level_m"].iloc[-1])
            last = float(block["water_level_m"].iloc[-1])
            label, color = risk_category(last)
            ttk.Label(f, text=s, font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=8, pady=(6,0))
            ttk.Label(f, text=f"Depletion: {dep:.2f} m").pack(anchor="w", padx=8)
            tk.Label(f, text=f"Risk: {label}", bg=color, fg="white", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=8, pady=(6,8))

        fig = Figure(figsize=(8,3), dpi=100); ax = fig.add_subplot(111)
        dep = (self.app.df.groupby("state").first()["water_level_m"] - self.app.df.groupby("state").last()["water_level_m"])
        ax.bar(dep.index, dep.values); ax.set_title("10-day Depletion by State"); ax.set_ylabel("Drop (m)"); ax.grid(True)
        FigureCanvasTkAgg(fig, master=self).get_tk_widget().pack(fill="x", padx=12, pady=10)

# Trends
class TrendsPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg="white")
        self.app = app
        self._build()

    def _build(self):
        top = tk.Frame(self, bg="white"); top.pack(fill="x", padx=12, pady=8)
        ttk.Label(top, text="Trends — Select state to show").pack(side="left")
        self.state_choice = ttk.Combobox(top, values=["All"] + sorted(self.app.df["state"].unique().tolist()), state="readonly", width=24)
        self.state_choice.current(0); self.state_choice.pack(side="left", padx=8)
        ttk.Button(top, text="Refresh", command=self.refresh).pack(side="left", padx=6)
        self.fig = Figure(figsize=(9,5), dpi=100); self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self); self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=6)
        self.refresh()

    def refresh(self):
        choice = self.state_choice.get()
        self.ax.clear()
        df = self.app.df if choice in ("All", "") else self.app.df[self.app.df["state"]==choice]
        for s, grp in df.groupby("state"):
            self.ax.plot(grp["timestamp"], grp["water_level_m"], marker="o", label=s)
        self.ax.set_title("Water Level Trends"); self.ax.set_ylabel("Water level (m)"); self.ax.grid(True); self.ax.legend()
        self.canvas.draw_idle()

# Simulator
class SimulatorPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg="white")
        self.app = app
        self._build()

    def _build(self):
        top = tk.Frame(self, bg="white"); top.pack(fill="x", padx=12, pady=8)
        ttk.Label(top, text="Simulator: Append a new simulated day (small decline)").pack(side="left")
        ttk.Button(top, text="Add simulated day", command=self.add_day).pack(side="left", padx=8)
        ttk.Button(top, text="Reset dataset", command=self.reset).pack(side="left", padx=6)
        self.fig = Figure(figsize=(9,4), dpi=100); self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self); self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=6)
        self.draw()

    def add_day(self):
        for s in self.app.df["state"].unique():
            block = self.app.df[self.app.df["state"]==s].sort_values("timestamp")
            last_date = block["timestamp"].iloc[-1]; last_val = block["water_level_m"].iloc[-1]
            decline = random.uniform(0.02, 0.06)
            new_row = {"timestamp": last_date + timedelta(days=1), "state": s, "location": block["location"].iloc[-1], "water_level_m": max(0.0, last_val - decline)}
            self.app.df = pd.concat([self.app.df, pd.DataFrame([new_row])], ignore_index=True)
        self.draw()

    def draw(self):
        self.ax.clear()
        for s, grp in self.app.df.sort_values("timestamp").groupby("state"):
            self.ax.plot(grp["timestamp"], grp["water_level_m"], marker="o", label=s)
        self.ax.set_title("Simulated Real-time Trends"); self.ax.grid(True); self.ax.legend(fontsize=8)
        self.canvas.draw_idle()

    def reset(self):
        self.app.df = load_groundwater_df()
        self.draw()
        messagebox.showinfo("Reset", "Dataset reset to original fixed data.")

# AI Center (with 5 models)
class AICenterPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg="white")
        self.app = app
        self._build()

    def _build(self):
        top = tk.Frame(self, bg="white"); top.pack(fill="x", padx=12, pady=8)
        ttk.Label(top, text="AI Center — Chat, Predict, Alerts, Recommend (5 Models)", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        body = tk.Frame(self, bg="white"); body.pack(fill="both", expand=True, padx=12, pady=6)
        left = tk.Frame(body, bg="white"); left.pack(side="left", fill="both", expand=True, padx=6)
        self.chat_log = tk.Text(left, height=18); self.chat_log.pack(fill="both", expand=True)
        self.chat_entry = ttk.Entry(left); self.chat_entry.pack(fill="x", pady=(6,0)); self.chat_entry.bind("<Return>", lambda e: self._on_chat())
        btns = tk.Frame(left); btns.pack(fill="x", pady=6)
        ttk.Button(btns, text="Send", command=self._on_chat).pack(side="left", padx=6)
        ttk.Button(btns, text="Clear", command=lambda: self.chat_log.delete("1.0", "end")).pack(side="left", padx=6)
        self._chat_write("GroundAI: Hello! Try 'predict Meghalaya 90' or 'recommend Punjab' or 'alerts 120'.")

        right = tk.Frame(body, bg="white", width=380); right.pack(side="left", fill="y", padx=12)
        ttk.Label(right, text="Select AI Model:").pack(anchor="w")
        self.model_cb = ttk.Combobox(right, state="readonly", values=[
            "Model 1 — Linear Forecast (60d)",
            "Model 2 — Danger Alerts (120d)",
            "Model 3 — Recommendations (all states)",
            "Model 4 — NLP Chat (use left pane)",
            "Model 5 — Long-term 10-Year Projection"
        ]); self.model_cb.current(0); self.model_cb.pack(fill="x", pady=6)
        ttk.Button(right, text="Run Model", command=self.run_model).pack(pady=6)
        self.ai_output = tk.Text(right, height=18); self.ai_output.pack(fill="both", pady=6)

    def _chat_write(self, t):
        self.chat_log.insert("end", t + "\n")
        self.chat_log.see("end")

    def _on_chat(self):
        msg = self.chat_entry.get().strip()
        if not msg: return
        self._chat_write("You: " + msg)
        ans = ai_chat_answer_smart(msg, self.app.df)
        self._chat_write("GroundAI: " + ans)
        self.chat_entry.delete(0, "end")

    def run_model(self):
        selection = self.model_cb.get()
        out = ""
        if "Linear Forecast" in selection:
            s = sorted(self.app.df["state"].unique())[0]
            sub = self.app.df[self.app.df["state"]==s]
            dates, preds, conf = forecast_enhanced(sub, days=60)
            if not dates:
                out = "Not enough data to forecast."
            else:
                out = f"60-day forecast for {s} (conf {conf*100:.0f}%):\n"
                for d,v in zip(dates[:12], preds[:12]):
                    out += f"{d.strftime('%Y-%m-%d')}: {v:.2f} m\n"
        elif "Danger Alerts" in selection:
            out = ""
            for s in sorted(self.app.df["state"].unique()):
                sub = self.app.df[self.app.df["state"]==s]
                alerts, conf = ai_alerts_for_state(sub, forecast_days=120)
                if alerts:
                    out += f"{s} alerts (conf {conf*100:.0f}%):\n"
                    for d,v in alerts[:5]:
                        out += f"- {d.strftime('%Y-%m-%d')}: {v:.2f} m\n"
            out = out or "No alerts in next 120 days."
        elif "Recommendations" in selection:
            out = ""
            for s in sorted(self.app.df["state"].unique()):
                sub = self.app.df[self.app.df["state"]==s]
                cur = float(sub["water_level_m"].iloc[-1])
                recs = ai_recommendation_for_state(cur) or ["No recommendation."]
                out += f"{s} (level {cur:.2f} m):\n - " + "\n - ".join(recs) + "\n\n"
        elif "NLP Chat" in selection:
            out = "Use the chat on the left. Type 'predict <state> <days>' or 'alerts <days>'."
        elif "10-Year" in selection:
            out = "10-year linear projection:\n"
            for s in sorted(self.app.df["state"].unique()):
                sub = self.app.df[self.app.df["state"]==s]
                dates, preds, conf = forecast_enhanced(sub, days=3650)
                if not dates:
                    out += f"{s}: insufficient history\n"
                else:
                    out += f"{s}: final (10 yr) = {preds[-1]:.2f} m (conf {conf*100:.0f}%)\n"
        else:
            out = "Model not recognized."

        self.ai_output.delete("1.0", "end")
        self.ai_output.insert("1.0", out)

# Data Table
class TablePage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg="white")
        self.app = app
        self._build()

    def _build(self):
        tk.Label(self, text="Data Table", font=("Segoe UI", 14, "bold"), bg="white").pack(anchor="w", padx=12, pady=8)
        cols = ("date","state","location","water_level")
        tv = ttk.Treeview(self, columns=cols, show="headings")
        for c in cols:
            tv.heading(c, text=c)
            tv.column(c, anchor="center", width=160)
        tv.pack(fill="both", expand=True, padx=12, pady=12)
        for _, r in self.app.df.iterrows():
            tv.insert("", "end", values=(r["timestamp"].strftime("%Y-%m-%d"), r["state"], r["location"], f"{r['water_level_m']:.2f}"))
# ReportsPage
class ReportsPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg="white")
        self.app = app
        self._build()

    def _build(self):
        tk.Label(self, text="Weekly Reports", font=("Segoe UI", 14, "bold"), bg="white").pack(anchor="w", padx=12, pady=8)
        container = tk.Frame(self, bg="white"); container.pack(fill="both", expand=True, padx=12, pady=6)
        left = tk.Frame(container, bg="white"); left.pack(side="left", fill="y", padx=8)
        tk.Label(left, text="Saved Reports", bg="white").pack(anchor="w")
        self.report_list = tk.Listbox(left, width=40, height=20); self.report_list.pack()
        btnf = tk.Frame(left, bg="white"); btnf.pack(pady=6)
        ttk.Button(btnf, text="Load", command=self.load_selected).pack(side="left", padx=6)
        ttk.Button(btnf, text="Refresh", command=self.refresh_list).pack(side="left", padx=6)
        ttk.Button(btnf, text="Delete", command=self.delete_selected).pack(side="left", padx=6)

        right = tk.Frame(container, bg="white"); right.pack(side="left", fill="both", expand=True, padx=8)
        ttk.Label(right, text="Title").pack(anchor="w"); self.title_entry = ttk.Entry(right); self.title_entry.pack(fill="x", pady=4)
        ttk.Label(right, text="Employee Name").pack(anchor="w"); self.employee_entry = ttk.Entry(right); self.employee_entry.pack(fill="x", pady=4); self.employee_entry.insert(0, self.app.current_user or "unknown")
        ttk.Label(right, text="Report Text").pack(anchor="w"); self.report_text = tk.Text(right, height=18); self.report_text.pack(fill="both", expand=True)
        btns = tk.Frame(right, bg="white"); btns.pack(pady=6)
        ttk.Button(btns, text="Import File", command=self.import_file).pack(side="left", padx=6)
        ttk.Button(btns, text="Save Report", command=self.save_report).pack(side="left", padx=6)
        ttk.Button(btns, text="Clear", command=lambda: self.report_text.delete("1.0","end")).pack(side="left", padx=6)
        self.refresh_list()

    def refresh_list(self):
        ensure_dir(REPORTS_DIR)
        self.report_list.delete(0, "end")
        for f in sorted(os.listdir(REPORTS_DIR)):
            if f.endswith(".txt"):
                self.report_list.insert("end", f)

    def save_report(self):
        title = self.title_entry.get().strip() or "Weekly_Report"
        employee = self.employee_entry.get().strip() or (self.app.current_user or "unknown")
        text = self.report_text.get("1.0", "end").strip()
        if not text:
            messagebox.showwarning("Empty", "Write the report before saving.")
            return
        ensure_dir(REPORTS_DIR)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"{timestamp}_{sanitize_filename(title)}.txt"
        path = os.path.join(REPORTS_DIR, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"Title: {title}\nEmployee: {employee}\nSaved: {timestamp}\n\n")
            f.write(text)
        messagebox.showinfo("Saved", f"Saved as {fname}"); self.refresh_list()

    def load_selected(self):
        sel = self.report_list.curselection()
        if not sel:
            messagebox.showwarning("Select", "Choose a report.")
            return
        name = self.report_list.get(sel[0])
        path = os.path.join(REPORTS_DIR, name)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        self.report_text.delete("1.0","end"); self.report_text.insert("1.0", content); messagebox.showinfo("Loaded", f"{name} loaded.")

    def delete_selected(self):
        sel = self.report_list.curselection()
        if not sel:
            messagebox.showwarning("Select", "Choose a report to delete.")
            return
        name = self.report_list.get(sel[0]); path = os.path.join(REPORTS_DIR, name)
        if messagebox.askyesno("Delete", f"Delete {name}?"):
            os.remove(path); self.refresh_list(); messagebox.showinfo("Deleted", f"{name} removed.")

    def import_file(self):
        file = filedialog.askopenfilename(title="Import", filetypes=[("Text","*.txt;*.md"),("CSV","*.csv"),("All","*.*")])
        if not file: return
        try:
            if file.lower().endswith(".csv"):
                df = pd.read_csv(file); content = df.to_string(index=False)
            else:
                with open(file, "r", encoding="utf-8", errors="ignore") as f: content = f.read()
        except Exception as e:
            messagebox.showerror("Import error", str(e)); return
        self.report_text.delete("1.0","end"); self.report_text.insert("1.0", content)
        suggested = os.path.splitext(os.path.basename(file))[0]
        self.title_entry.delete(0,"end"); self.title_entry.insert(0, suggested)
        messagebox.showinfo("Imported", f"Imported {os.path.basename(file)}")

# Smart Alerts page — table summary + refresh
class SmartAlertsPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg="white")
        self.app = app
        self._build()

    def _build(self):
        top = tk.Frame(self, bg="white"); top.pack(fill="x", padx=12, pady=8)
        ttk.Label(top, text="Smart Alerts Dashboard", font=("Segoe UI", 14, "bold")).pack(side="left")
        ttk.Label(top, text="Forecast horizon (days):").pack(side="left", padx=8)
        self.horizon = ttk.Spinbox(top, from_=7, to=3650, width=8); self.horizon.set(90); self.horizon.pack(side="left")
        ttk.Button(top, text="Refresh Alerts", command=self.refresh).pack(side="left", padx=8)

        cols = ("State","Current","Projected","Risk")
        self.tv = ttk.Treeview(self, columns=cols, show="headings", height=15)
        for c in cols:
            self.tv.heading(c, text=c); self.tv.column(c, anchor="center", width=180)
        self.tv.pack(fill="both", expand=True, padx=12, pady=12)
        self.refresh()

    def refresh(self):
        for i in self.tv.get_children(): self.tv.delete(i)
        days = int(self.horizon.get())
        for s in sorted(self.app.df["state"].unique()):
            sub = self.app.df[self.app.df["state"]==s].sort_values("timestamp")
            current = float(sub["water_level_m"].iloc[-1])
            dates, preds, conf = forecast_enhanced(sub, days=days)
            proj = preds[-1] if preds else current
            label, _ = risk_category(proj)
            self.tv.insert("", "end", values=(s, f"{current:.2f} m", f"{proj:.2f} m", label))

# Recharge Calculator — advanced
class RechargeCalculatorPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg="white")
        self.app = app
        self._build()

    def _build(self):
        tk.Label(self, text="Recharge Calculator — Estimate Safe Recharge Volume", font=("Segoe UI", 14, "bold"), bg="white").pack(anchor="w", padx=12, pady=8)
        frm = tk.Frame(self, bg="white"); frm.pack(padx=12, pady=8, anchor="w")
        # inputs: catchment area, avg rainfall (mm), runoff coeff, infiltration %
        tk.Label(frm, text="Catchment area (sq.m)").grid(row=0, column=0, sticky="w", pady=6)
        self.area_e = ttk.Entry(frm); self.area_e.grid(row=0, column=1, padx=8)
        tk.Label(frm, text="Average annual rainfall (mm)").grid(row=1, column=0, sticky="w", pady=6)
        self.rain_e = ttk.Entry(frm); self.rain_e.grid(row=1, column=1, padx=8)
        tk.Label(frm, text="Runoff coefficient (0-1)").grid(row=2, column=0, sticky="w", pady=6)
        self.runoff_e = ttk.Entry(frm); self.runoff_e.grid(row=2, column=1, padx=8); self.runoff_e.insert(0,"0.4")
        tk.Label(frm, text="Infiltration efficiency (%)").grid(row=3, column=0, sticky="w", pady=6)
        self.infil_e = ttk.Entry(frm); self.infil_e.grid(row=3, column=1, padx=8); self.infil_e.insert(0,"50")
        tk.Label(frm, text="Desired recharge (m3)").grid(row=4, column=0, sticky="w", pady=6)
        self.recharge_target_e = ttk.Entry(frm); self.recharge_target_e.grid(row=4, column=1, padx=8)
        ttk.Button(frm, text="Calculate", command=self.calculate).grid(row=5, column=0, columnspan=2, pady=10)
        # results
        self.result_text = tk.Text(self, height=12); self.result_text.pack(fill="both", padx=12, pady=8)

    def calculate(self):
        try:
            area = float(self.area_e.get())  # m^2
            rainfall_mm = float(self.rain_e.get())
            runoff = float(self.runoff_e.get())
            infil_pct = float(self.infil_e.get())/100.0
            target = self.recharge_target_e.get().strip()
            # compute annual runoff volume (m3) = area * rainfall_m * (1 - runoff coeff?) careful:
            # Rainfall volume = area * rainfall_m. Collectable volume ~ area * rainfall_m * (1 - runoff)
            rainfall_m = rainfall_mm / 1000.0
            total_rain_vol = area * rainfall_m
            collectable = total_rain_vol * (1 - runoff) * infil_pct
            # suggested pit sizing (simple): pit depth 2 m, volume per pit = area_pit * depth => area_pit = desired_volume / depth
            suggested_depth = 2.0
            pit_area = collectable / suggested_depth if suggested_depth>0 else 0.0
            # if user typed desired recharge compute number of pits needed
            out = ""
            out += f"Total annual rainfall volume on catchment: {total_rain_vol:,.2f} m³\n"
            out += f"Estimated collectable (after runoff & infiltration): {collectable:,.2f} m³ (infil % applied)\n"
            if target:
                try:
                    desired = float(target)
                    pits_needed = desired / (suggested_depth * pit_area) if pit_area>0 else (desired / (suggested_depth*10))
                    # simpler: volume per pit = suggested_depth * pit_area (but pit_area computed from collectable ¯\_(ツ)_/¯) -> instead recommend pit area:
                    pit_area_for_target = desired / suggested_depth
                    out += f"\nTo achieve desired recharge {desired:,.2f} m³:\n - Required pit area (depth {suggested_depth} m): {pit_area_for_target:,.2f} m²\n"
                except:
                    pass
            out += f"\nSuggested single pit depth: {suggested_depth} m\nSuggested pit area (if using full annual collectable): {pit_area:,.2f} m²\n"
            # recommendations
            out += "\nRecommendations:\n"
            if collectable < 50:
                out += "- Low collectable volume: consider rooftop harvesting & multiple small pits.\n"
            elif collectable < 500:
                out += "- Moderate collectable: install 1-3 recharge pits and manage runoff.\n"
            else:
                out += "- Large volume: plan check dams, recharge basins and regulatory measures.\n"
            self.result_text.delete("1.0","end"); self.result_text.insert("1.0", out)
        except Exception as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

# About page
class AboutPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg="white")
        self.app = app
        self._build()

    def _build(self):
        tk.Label(self, text="About — GW App (Version A)", font=("Segoe UI", 14, "bold"), bg="white").pack(anchor="w", padx=12, pady=8)
        self.txt = tk.Text(self, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=12, pady=8)
        content = ""
        if os.path.exists(ABOUT_FILE):
            with open(ABOUT_FILE, "r", encoding="utf-8") as f:
                content = f.read()
        else:
            content = "GW App — AI enhanced groundwater dashboard.\n\nFeatures:\n- Trends, Simulator, AI Center\n- Recharge Calculator\n- Smart Alerts\n- Weekly reports\n"
        self.txt.insert("1.0", content); self.txt.config(state="disabled")
        bf = tk.Frame(self); bf.pack(pady=8)
        ttk.Button(bf, text="Edit", command=self.enable_edit).pack(side="left", padx=6)
        ttk.Button(bf, text="Save", command=self.save_about).pack(side="left", padx=6)

    def enable_edit(self):
        self.txt.config(state="normal"); messagebox.showinfo("Edit", "Now editable. Click Save to persist.")

    def save_about(self):
        content = self.txt.get("1.0", "end").strip()
        with open(ABOUT_FILE, "w", encoding="utf-8") as f:
            f.write(content)
        self.txt.config(state="disabled")
        messagebox.showinfo("Saved", "About page saved.")

# Export helpers
def export_dataset_csv(df):
    ensure_dir(EXPORT_DIR)
    path = os.path.join(EXPORT_DIR, f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(path, index=False)
    return path

def export_forecast_csv(df, days=30):
    ensure_dir(EXPORT_DIR)
    rows = []
    for s in sorted(df["state"].unique()):
        sub = df[df["state"]==s].sort_values("timestamp")
        dates, preds, conf = forecast_enhanced(sub, days=days)
        for d, p in zip(dates, preds):
            rows.append({"state": s, "date": d.strftime("%Y-%m-%d"), "predicted_level_m": p, "confidence": conf})
    out_df = pd.DataFrame(rows)
    path = os.path.join(EXPORT_DIR, f"forecast_{days}d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    out_df.to_csv(path, index=False)
    return path

def export_all_zip(df):
    ensure_dir(EXPORT_DIR)
    zipname = os.path.join(EXPORT_DIR, f"gw_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip")
    with zipfile.ZipFile(zipname, "w", compression=zipfile.ZIP_DEFLATED) as z:
        ds_path = export_dataset_csv(df)
        fc_path = export_forecast_csv(df, days=30)
        z.write(ds_path, arcname=os.path.basename(ds_path))
        z.write(fc_path, arcname=os.path.basename(fc_path))
        if os.path.exists(REPORTS_DIR):
            for fname in os.listdir(REPORTS_DIR):
                z.write(os.path.join(REPORTS_DIR, fname), arcname=os.path.join("reports", fname))
        if os.path.exists(ABOUT_FILE):
            z.write(ABOUT_FILE, arcname="about_app.txt")
    return zipname

# Settings / Admin page
class SettingsPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg="white")
        self.app = app
        self._build()

    def _build(self):
        tk.Label(self, text="Settings & Admin", font=("Segoe UI", 14, "bold"), bg="white").pack(anchor="w", padx=12, pady=8)
        admin_frame = tk.LabelFrame(self, text="User Management", padx=12, pady=12)
        admin_frame.pack(fill="x", padx=12, pady=8)
        cols = ("username","role")
        self.tv = ttk.Treeview(admin_frame, columns=cols, show="headings", height=6)
        for c in cols: self.tv.heading(c, text=c); self.tv.column(c, width=160)
        self.tv.pack(fill="x", pady=6)
        btnf = tk.Frame(admin_frame); btnf.pack(pady=6)
        ttk.Button(btnf, text="Refresh", command=self.refresh_users).pack(side="left", padx=6)
        ttk.Button(btnf, text="Add User", command=self.add_user).pack(side="left", padx=6)
        ttk.Button(btnf, text="Edit Password", command=self.edit_password).pack(side="left", padx=6)
        ttk.Button(btnf, text="Delete User", command=self.delete_user).pack(side="left", padx=6)

        actions = tk.LabelFrame(self, text="Admin Actions", padx=12, pady=12)
        actions.pack(fill="x", padx=12, pady=8)
        ttk.Button(actions, text="Reset dataset", command=self.reset_dataset).pack(side="left", padx=6)
        ttk.Button(actions, text="Delete all reports", command=self.delete_reports).pack(side="left", padx=6)
        ttk.Button(actions, text="Export dataset (CSV)", command=self.export_dataset).pack(side="left", padx=6)
        ttk.Button(actions, text="Export forecast (CSV)", command=self.export_forecast).pack(side="left", padx=6)
        ttk.Button(actions, text="Export all (ZIP)", command=self.export_all).pack(side="left", padx=6)
        self.refresh_users()

    def refresh_users(self):
        for i in self.tv.get_children(): self.tv.delete(i)
        users = load_users()
        for u, info in users.items():
            role = info.get("role", "user") if isinstance(info, dict) else "user"
            self.tv.insert("", "end", values=(u, role))

    def add_user(self):
        dlg = tk.Toplevel(self); dlg.title("Add User"); dlg.geometry("360x220")
        ttk.Label(dlg, text="Username").pack(pady=(12,0)); e_u = ttk.Entry(dlg); e_u.pack(pady=6)
        ttk.Label(dlg, text="Password").pack(pady=(8,0)); e_p = ttk.Entry(dlg, show="•"); e_p.pack(pady=6)
        ttk.Label(dlg, text="Role").pack(pady=(8,0)); e_r = ttk.Entry(dlg); e_r.insert(0,"user"); e_r.pack(pady=6)
        def create():
            u = e_u.get().strip(); p = e_p.get().strip(); r = e_r.get().strip() or "user"
            if not u or not p: messagebox.showwarning("Missing","Enter username & password"); return
            users = load_users()
            if u in users: messagebox.showerror("Exists","User exists"); return
            users[u] = {"password": p, "role": r}; save_users(users); dlg.destroy(); self.refresh_users(); messagebox.showinfo("Added","User added")
        ttk.Button(dlg, text="Create", command=create).pack(pady=12)

    def edit_password(self):
        sel = self.tv.selection()
        if not sel: messagebox.showwarning("Select","Select user"); return
        username = self.tv.item(sel[0])["values"][0]
        dlg = tk.Toplevel(self); dlg.title("Edit Password"); dlg.geometry("360x180")
        ttk.Label(dlg, text=f"New password for {username}").pack(pady=(12,0)); e_p = ttk.Entry(dlg, show="•"); e_p.pack(pady=8)
        def save_pw():
            p = e_p.get().strip(); users = load_users(); users[username]["password"] = p; save_users(users); dlg.destroy(); self.refresh_users(); messagebox.showinfo("Saved","Password updated")
        ttk.Button(dlg, text="Save", command=save_pw).pack(pady=8)

    def delete_user(self):
        sel = self.tv.selection(); 
        if not sel: messagebox.showwarning("Select","Select user"); return
        username = self.tv.item(sel[0])["values"][0]
        if messagebox.askyesno("Delete", f"Delete {username}?"):
            users = load_users(); users.pop(username, None); save_users(users); self.refresh_users(); messagebox.showinfo("Deleted","User removed")

    def reset_dataset(self):
        if messagebox.askyesno("Reset","Reset dataset to default?"):
            self.app.df = load_groundwater_df(); messagebox.showinfo("Reset","Dataset reset")

    def delete_reports(self):
        if messagebox.askyesno("Delete","Delete all reports?"):
            if os.path.exists(REPORTS_DIR): shutil.rmtree(REPORTS_DIR)
            messagebox.showinfo("Deleted","All reports deleted")

    def export_dataset(self):
        path = export_dataset_csv(self.app.df); messagebox.showinfo("Exported", f"CSV exported to:\n{path}")

    def export_forecast(self):
        path = export_forecast_csv(self.app.df, days=30); messagebox.showinfo("Exported", f"Forecast exported to:\n{path}")

    def export_all(self):
        path = export_all_zip(self.app.df); messagebox.showinfo("Exported", f"ZIP exported to:\n{path}")

# ----------------- Start the app -----------------
if __name__ == "__main__":
    app = GroundwaterApp()
    try:
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    app.mainloop()

