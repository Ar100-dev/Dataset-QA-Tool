"""
gui.py  —  Dataset QA Tool V2.2
Main Tkinter application window.

Changes from V2.1:
    • Imports get_splits() from checks — no split names hardcoded anywhere in GUI.
    • validate_dataset_structure() now returns (is_valid, splits, problems).
    • All check functions receive discovered splits list.
    • Stat cards generated dynamically from whatever splits are found
      (train-only, train+val, train+val+test, any custom name).
    • Fixed-key stat cards (train_images etc.) replaced with dynamic per-split
      rows plus always-visible Total Images / Total Labels cards.
    • View Log toggle, dark theme, metric cards, issue window — all preserved.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from checks import (
    validate_dataset_structure,
    detect_layout,
    get_splits,
    get_image_label_counts,
    check_missing_pairs,
    check_corrupt_images,
    check_duplicates,
    validate_labels,
    execute_auto_fixes,
    LAYOUT_A, LAYOUT_B,
)
from analyzer import class_distribution


# ── Colour palette ─────────────────────────────────────────────────────────
BG       = "#0F172A"
PANEL    = "#1E293B"
BORDER   = "#334155"

ACCENT   = "#3B82F6"
ACCENT2  = "#60A5FA"

SUCCESS  = "#22C55E"
WARNING  = "#F59E0B"
DANGER   = "#EF4444"

TEXT      = "#F8FAFC"
TEXT_DIM  = "#94A3B8"

ENTRY_BG = "#172033"

# Cycle of colours for per-split stat cards
SPLIT_COLORS = [ACCENT, ACCENT2, "#38BDF8", "#F472B6", "#FB923C", "#A3E635"]


class DatasetQATool(tk.Tk):

    def __init__(self) -> None:
        super().__init__()
        self.title("Dataset QA Tool  V2")
        self.geometry("1500x920")
        self.minsize(1000, 780)
        self.configure(bg=BG)

        self.dataset_path: str = ""
        self.issue_details: dict[str, list[str]] = {}
        self.graph_canvas: FigureCanvasTkAgg | None = None
        self._log_visible: bool = True

        self._apply_style()
        self._build_ui()

    # ── ttk Style ──────────────────────────────────────────────────────────
    def _apply_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".",
            background=BG, foreground=TEXT,
            fieldbackground=ENTRY_BG, bordercolor=BORDER,
            troughcolor=PANEL, selectbackground=ACCENT,
            selectforeground=TEXT, font=("Segoe UI", 10))
        style.configure("TLabelframe",
            background=PANEL, relief="flat", bordercolor=BORDER, borderwidth=1)
        style.configure("TLabelframe.Label",
            background=PANEL, foreground=ACCENT2,
            font=("Segoe UI", 10, "bold"))
        style.configure("TEntry",
            fieldbackground=ENTRY_BG, foreground=TEXT,
            bordercolor=BORDER, insertcolor=TEXT)
        style.configure("TButton",
            background=PANEL, foreground=TEXT,
            bordercolor=BORDER, relief="flat",
            padding=(12, 6), font=("Segoe UI", 10))
        style.map("TButton",
            background=[("active", BORDER), ("pressed", ACCENT)],
            foreground=[("active", TEXT)])
        style.configure("Primary.TButton",
            background=ACCENT, foreground="#FFFFFF",
            bordercolor=ACCENT, font=("Segoe UI", 10, "bold"), padding=(20,10))
        style.map("Primary.TButton",
            background=[("active", "#4A7AE0"), ("pressed", "#3A6AD0")])
        style.configure("Success.TButton",
            background="#1E4A3A", foreground=SUCCESS,
            bordercolor="#2A6A52", font=("Segoe UI", 10), padding=(18, 6))
        style.map("Success.TButton",
            background=[("active", "#2A6A52"), ("pressed", "#1A3A2A")])
        style.configure("Warning.TButton",
            background="#3A2E10", foreground=WARNING,
            bordercolor="#5A4A18", font=("Segoe UI", 10), padding=(12, 6))
        style.map("Warning.TButton",
            background=[("active", "#5A4A18"), ("pressed", "#2A2008")])
        style.configure("TProgressbar",
            troughcolor=PANEL, background=ACCENT,
            bordercolor=BORDER, thickness=8)
        style.configure("TScrollbar",
            background=PANEL, troughcolor=BG,
            bordercolor=BORDER, arrowcolor=TEXT_DIM)

    # ── UI Construction ────────────────────────────────────────────────────
    def _build_ui(self) -> None:

        # Header
        header = tk.Frame(self, bg=PANEL, height=75)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="  ◈  Dataset QA Tool",
                 font=("Segoe UI", 22, "bold"), bg=PANEL, fg=ACCENT
                 ).pack(side="left", padx=16, pady=10)
        tk.Label(header, text="V2   •   YOLO Format",
                 font=("Segoe UI", 9), bg=PANEL, fg=TEXT_DIM
                 ).pack(side="left", pady=10)

        # Path row
        path_outer = tk.Frame(self, bg=BG)
        path_outer.pack(fill="x", padx=14, pady=(10, 4))
        tk.Label(path_outer, text="Dataset Path",
                 font=("Segoe UI", 9), bg=BG, fg=TEXT_DIM).pack(anchor="w")
        path_row = tk.Frame(path_outer, bg=BG)
        path_row.pack(fill="x", pady=(2, 0))
        self.path_var = tk.StringVar()
        tk.Entry(path_row, textvariable=self.path_var,
                 bg=ENTRY_BG, fg=TEXT, insertbackground=TEXT,
                 relief="flat", font=("Segoe UI", 11),
                 highlightthickness=1, highlightbackground=BORDER,
                 highlightcolor=ACCENT,
                 ).pack(side="left", fill="x", expand=True, ipady=10)
        ttk.Button(path_row, text="  Browse…",
                   command=self._browse_dataset).pack(side="left", padx=(8, 0))

        # Button bar
        self._btn_frame = tk.Frame(self, bg=BG)
        self._btn_frame.pack(pady=8)
        ttk.Button(self._btn_frame, text="▶  Run Verification",
                   style="Primary.TButton",
                   command=self._start_verification).pack(side="left", padx=5)
        self.show_graph_btn = ttk.Button(self._btn_frame, text="◆  Show Graph",
                   style="Success.TButton", command=self._show_graph_clicked)
        self.view_issues_btn = ttk.Button(self._btn_frame, text="⚑  View Issues",
                   style="Warning.TButton", command=self._open_issues_window)
        self.auto_fix_btn = ttk.Button(
            self._btn_frame,
            text="🛠 Auto Fix",
            style="Warning.TButton",
            command=self._auto_fix
        )
        self.view_log_btn = ttk.Button(self._btn_frame, text="☰  Hide Log",
                   command=self._toggle_log)
        self.view_log_btn.pack(side="left", padx=5)

        # Progress bar
        prog_outer = tk.Frame(self, bg=BG)
        prog_outer.pack(fill="x", padx=14, pady=(0, 6))
        self.progress = ttk.Progressbar(prog_outer, mode="determinate")
        self.progress.pack(fill="x")
        self._prog_label = tk.Label(prog_outer, text="",
                                    font=("Segoe UI", 8), bg=BG, fg=TEXT_DIM, anchor="e")
        self._prog_label.pack(fill="x")

        # Main content row
        content = tk.Frame(self, bg=BG)
        content.pack(fill="both", expand=True, padx=14, pady=2)

        # Left column — stats + results
        left_col = tk.Frame(content, bg=BG, width=260)
        left_col.pack(side="left", fill="y", padx=(0, 10))
        left_col.pack_propagate(False)

        # ── Dataset Statistics header + scrollable card area ───────────────
        tk.Label(left_col, text="Dataset Statistics",
                 font=("Segoe UI", 9, "bold"), bg=BG, fg=ACCENT2
                 ).pack(anchor="w", pady=(0, 4))

        # Container that will be rebuilt after each verification
        self._stats_container = tk.Frame(left_col, bg=BG)
        self._stats_container.pack(fill="x")

        # Placeholder cards (2 totals only, before first run)
        self._total_img_var = tk.StringVar(value="—")
        self._total_lbl_var = tk.StringVar(value="—")
        self._render_initial_stat_cards()

        # Verification Results
        tk.Label(left_col, text="Verification Results",
                 font=("Segoe UI", 9, "bold"), bg=BG, fg=ACCENT2
                 ).pack(anchor="w", pady=(12, 4))

        self._result_vars: dict[str, tk.StringVar] = {}
        checks_meta = [
            ("Missing Labels",   "missing_labels",   DANGER),
            ("Missing Images",   "missing_images",   DANGER),
            ("Corrupt Images",   "corrupt_images",   WARNING),
            ("Duplicate Images", "duplicate_images", WARNING),
            ("Invalid Labels",   "invalid_labels",   DANGER),
            ("Empty Labels",     "empty_labels",     TEXT_DIM),
        ]
        results_frame = tk.Frame(left_col, bg=PANEL,
                                 highlightthickness=1, highlightbackground=BORDER,
                                 padx=10, pady=8)
        results_frame.pack(fill="x")
        for label, key, color in checks_meta:
            var = tk.StringVar(value="—")
            self._result_vars[key] = var
            row = tk.Frame(results_frame, bg=PANEL)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, font=("Segoe UI", 10),
                     bg=PANEL, fg=TEXT, width=18, anchor="w").pack(side="left")
            tk.Label(row, textvariable=var, font=("Segoe UI", 10, "bold"),
                     bg=PANEL, fg=color, width=6, anchor="e").pack(side="right")

        # Right column — graph
        right_col = tk.Frame(content, bg=PANEL,
                             highlightthickness=1, highlightbackground=BORDER)
        right_col.pack(side="left", fill="both", expand=True)
        self._graph_outer = right_col
        self._graph_placeholder = tk.Label(
            right_col,
            text="Run verification, then click\n◆  Show Graph",
            font=("Segoe UI", 11), bg=PANEL, fg=TEXT_DIM)
        self._graph_placeholder.pack(expand=True)

        # Log panel
        self._log_frame = tk.Frame(self, bg=PANEL,
                                   highlightthickness=1, highlightbackground=BORDER)
        self._log_frame.pack(fill="both", expand=False, padx=14, pady=(6, 10))
        tk.Label(self._log_frame, text="Logs",
                 font=("Segoe UI", 9, "bold"), bg=PANEL, fg=ACCENT2
                 ).pack(anchor="w", padx=8, pady=(6, 0))
        log_body = tk.Frame(self._log_frame, bg=PANEL)
        log_body.pack(fill="both", expand=True)
        self.log_text = tk.Text(log_body, height=10, wrap="word",
                                bg=BG, fg=SUCCESS, font=("Consolas", 9),
                                insertbackground=TEXT, relief="flat",
                                padx=8, pady=4)
        log_scroll = ttk.Scrollbar(log_body, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")
        self.log_text.tag_configure("ok",   foreground=SUCCESS)
        self.log_text.tag_configure("err",  foreground=DANGER)
        self.log_text.tag_configure("warn", foreground=WARNING)
        self.log_text.tag_configure("dim",  foreground=TEXT_DIM)
        self.log_text.tag_configure("head", foreground=ACCENT)

    # ── Initial / placeholder stat cards ───────────────────────────────────
    def _render_initial_stat_cards(self) -> None:
        """Show two placeholder Total cards before first verification."""
        for w in self._stats_container.winfo_children():
            w.destroy()
        grid = tk.Frame(self._stats_container, bg=BG)
        grid.pack(fill="x")
        for col, (lbl, var, color) in enumerate([
            ("Total Images", self._total_img_var, SUCCESS),
            ("Total Labels", self._total_lbl_var, WARNING),
        ]):
            card = tk.Frame(grid, bg=PANEL, padx=12, pady=8,
                            highlightthickness=1, highlightbackground=BORDER)
            card.grid(row=0, column=col, padx=4, pady=4, sticky="nsew")
            tk.Label(card, text=lbl, font=("Segoe UI", 8),
                     bg=PANEL, fg=TEXT_DIM).pack(anchor="w")
            tk.Label(card, textvariable=var, font=("Segoe UI", 18, "bold"),
                     bg=PANEL, fg=color).pack(anchor="w")
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

    # ── Dynamic stat cards (called after verification) ──────────────────
    def _render_stat_cards(self, stats: dict, splits: list[str]) -> None:
        """
        Rebuild stat cards dynamically based on actual splits found.
        Always shows per-split image+label counts, plus Total row.
        """
        for w in self._stats_container.winfo_children():
            w.destroy()

        grid = tk.Frame(self._stats_container, bg=BG)
        grid.pack(fill="x")

        row_idx = 0
        for i, split in enumerate(splits):
            color = SPLIT_COLORS[i % len(SPLIT_COLORS)]
            for col, (suffix, kind) in enumerate([("_images", "Images"), ("_labels", "Labels")]):
                key = split + suffix
                val = str(stats.get(key, 0))
                card = tk.Frame(grid, bg=PANEL, padx=10, pady=6,
                                highlightthickness=1, highlightbackground=BORDER)
                card.grid(row=row_idx, column=col, padx=4, pady=3, sticky="nsew")
                tk.Label(card, text=f"{split.capitalize()} {kind}",
                         font=("Segoe UI", 8), bg=PANEL, fg=TEXT_DIM).pack(anchor="w")
                tk.Label(card, text=val, font=("Segoe UI", 16, "bold"),
                         bg=PANEL, fg=color).pack(anchor="w")
            row_idx += 1

        # Total cards
        for col, (lbl, key, color) in enumerate([
            ("Total Images", "total_images", SUCCESS),
            ("Total Labels", "total_labels", WARNING),
        ]):
            card = tk.Frame(grid, bg=PANEL, padx=10, pady=6,
                            highlightthickness=1, highlightbackground=BORDER)
            card.grid(row=row_idx, column=col, padx=4, pady=3, sticky="nsew")
            tk.Label(card, text=lbl, font=("Segoe UI", 8),
                     bg=PANEL, fg=TEXT_DIM).pack(anchor="w")
            tk.Label(card, text=str(stats.get(key, 0)),
                     font=("Segoe UI", 16, "bold"), bg=PANEL, fg=color).pack(anchor="w")

        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

    # ── Browse ─────────────────────────────────────────────────────────────
    def _browse_dataset(self) -> None:
        folder = filedialog.askdirectory()
        if folder:
            self.dataset_path = folder
            self.path_var.set(folder)

    # ── Start verification ─────────────────────────────────────────────────
    def _start_verification(self) -> None:
        if not self.dataset_path:
            messagebox.showerror("Error", "Select a dataset folder first.")
            return
        self.show_graph_btn.pack_forget()
        self.view_issues_btn.pack_forget()
        threading.Thread(target=self._run_verification, daemon=True).start()

    # ── Worker ────────────────────────────────────────────────────────────
    def _run_verification(self) -> None:
        try:
            start = time.time()
            self._safe_after(self._reset_results)
            self._log_safe("──────────────────────────────────", "head")
            self._log_safe("Verification Started…\n", "head")
            self._set_progress(0, "Starting…")

            # ── Structure check ───────────────────────────────────────────
            self._log_safe("Checking dataset structure…", "dim")
            is_valid, splits, problems = validate_dataset_structure(self.dataset_path)

            if not is_valid:
                msg = "Invalid Dataset Structure\n\n" + "\n".join(f"  • {p}" for p in problems)
                self._log_safe("✗ Structure problems found:", "err")
                for p in problems:
                    self._log_safe(f"    • {p}", "err")
                self._log_safe("\nVerification aborted.", "err")
                self._safe_after(lambda: messagebox.showerror("Structure Error", msg))
                return

            layout = detect_layout(self.dataset_path)
            layout_name = "Standard (images/<split>/)" if layout == LAYOUT_A else "Roboflow (<split>/images/)"
            self._log_safe(f"✓ Layout detected: {layout_name}", "ok")
            self._log_safe(f"✓ Splits found: {', '.join(splits)}\n", "ok")
            self._set_progress(10, f"Layout: {layout_name} | Splits: {', '.join(splits)}")

            # ── Step 1: Counts ────────────────────────────────────────────
            self._log_safe("Checking image and label counts…", "dim")
            stats = get_image_label_counts(self.dataset_path, splits)
            for sp in splits:
                self._log_safe(
                    f"  {sp}: {stats[sp+'_images']} images, "
                    f"{stats[sp+'_labels']} labels", "dim")
            self._log_safe("✓ Image count check completed\n", "ok")
            self._set_progress(25, "Counted images & labels")

            # ── Step 2: Missing pairs ─────────────────────────────────────
            self._log_safe("Checking missing image-label pairs…", "dim")
            ml_count, mi_count, ml_files, mi_files = check_missing_pairs(
                self.dataset_path, splits)
            self._log_safe("✓ Missing pair check completed\n", "ok")
            self._set_progress(45, "Checked missing pairs")

            # ── Step 3: Corrupt ───────────────────────────────────────────
            self._log_safe("Checking corrupt images…", "dim")
            corrupt_count, corrupt_files = check_corrupt_images(self.dataset_path, splits)
            self._log_safe("✓ Corrupt image check completed\n", "ok")
            self._set_progress(60, "Checked corrupt images")

            # ── Step 4: Duplicates ────────────────────────────────────────
            self._log_safe("Checking duplicate images…", "dim")
            dup_count, dup_files = check_duplicates(self.dataset_path, splits)
            self._log_safe("✓ Duplicate image check completed\n", "ok")
            self._set_progress(75, "Checked duplicates")

            # ── Step 5: Labels ────────────────────────────────────────────
            self._log_safe("Checking label validity…", "dim")
            inv_count, emp_count, inv_files, emp_files = validate_labels(
                self.dataset_path, splits)
            self._log_safe("✓ Label validation completed\n", "ok")
            self._set_progress(88, "Validated labels")

            # ── Build issue details ───────────────────────────────────────
            self._log_safe("Generating final results…", "dim")
            issue_details = {
                "Missing Labels":   ml_files,
                "Missing Images":   mi_files,
                "Corrupt Images":   corrupt_files,
                "Duplicate Images": dup_files,
                "Invalid Labels":   inv_files,
                "Empty Labels":     emp_files,
            }

            # ── Update UI on main thread ──────────────────────────────────
            self._safe_after(lambda s=stats, sp=splits,
                                    ml=ml_count, mi=mi_count, cc=corrupt_count,
                                    dc=dup_count, ic=inv_count, ec=emp_count,
                                    id_=issue_details:
                self._update_panels(s, sp, ml, mi, cc, dc, ic, ec, id_))

            self._log_safe("✓ Results generated\n", "ok")
            elapsed = round(time.time() - start, 2)
            self._set_progress(100, f"Done in {elapsed}s")
            self._log_safe(f"Completed in {elapsed} sec", "head")
            self._safe_after(self._on_verification_success)

        except Exception:
            tb = traceback.format_exc()
            print(tb)
            self._log_safe("\n══ ERROR OCCURRED ══", "err")
            self._log_safe(tb, "err")

    # ── Thread-safe helpers ────────────────────────────────────────────────
    def _safe_after(self, fn) -> None:
        self.after(0, fn)

    def _log_safe(self, text: str, tag: str = "") -> None:
        self.after(0, lambda t=text, tg=tag: self._append_log(t, tg))

    def _set_progress(self, value: int, label: str = "") -> None:
        self.after(0, lambda v=value, l=label: (
            self.progress.configure(value=v),
            self._prog_label.configure(text=l),
        ))

    def _append_log(self, text: str, tag: str = "") -> None:
        self.log_text.insert(tk.END, text + "\n", tag if tag else ())
        self.log_text.see(tk.END)

    def _reset_results(self) -> None:
        for var in self._result_vars.values():
            var.set("—")

    def _update_panels(
        self, stats, splits,
        ml_count, mi_count, corrupt_count,
        dup_count, inv_count, emp_count,
        issue_details,
    ) -> None:
        self.issue_details = issue_details

        # Rebuild stat cards for discovered splits
        self._render_stat_cards(stats, splits)

        # Update result rows
        self._result_vars["missing_labels"].set(str(ml_count))
        self._result_vars["missing_images"].set(str(mi_count))
        self._result_vars["corrupt_images"].set(str(corrupt_count))
        self._result_vars["duplicate_images"].set(str(dup_count))
        self._result_vars["invalid_labels"].set(str(inv_count))
        self._result_vars["empty_labels"].set(str(emp_count))

    def _on_verification_success(self) -> None:
        self.show_graph_btn.pack(side="left", padx=5)
        self.view_issues_btn.pack(side="left", padx=5)
        self.auto_fix_btn.pack(side="left", padx=5)

    # ── Show Graph ─────────────────────────────────────────────────────────
    def _show_graph_clicked(self) -> None:
        self._render_graph()

    def _render_graph(self) -> None:
        data = class_distribution(self.dataset_path)
        if not data:
            self._append_log("No class distribution data found.", "warn")
            return

        if self.graph_canvas:
            self.graph_canvas.get_tk_widget().destroy()
            self.graph_canvas = None

        if self._graph_placeholder.winfo_exists():
            self._graph_placeholder.destroy()

        classes = list(data.keys())
        counts  = list(data.values())

        plt.rcParams.update({
            "figure.facecolor": PANEL, "axes.facecolor": PANEL,
            "axes.edgecolor": BORDER, "axes.labelcolor": TEXT_DIM,
            "xtick.color": TEXT_DIM, "ytick.color": TEXT_DIM,
            "text.color": TEXT, "grid.color": BORDER,
            "grid.linestyle": "--", "grid.alpha": 0.5,
        })

        fig, ax = plt.subplots(figsize=(7, 3.8))
        bars = ax.bar(classes, counts, color=ACCENT, edgecolor=PANEL, linewidth=0.8)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts) * 0.015,
                    str(count), ha="center", va="bottom",
                    fontsize=8, color=TEXT)
        ax.set_title("Class Distribution", fontsize=12, fontweight="bold",
                     color=TEXT, pad=10)
        ax.set_xlabel("Object Classes", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_color(BORDER)
        ax.spines["bottom"].set_color(BORDER)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
        plt.xticks(rotation=40, ha="right", fontsize=8)
        fig.tight_layout()

        self.graph_canvas = FigureCanvasTkAgg(fig, master=self._graph_outer)
        self.graph_canvas.draw()
        self.graph_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.graph_canvas.get_tk_widget().configure(bg=PANEL)
        self._append_log("✓ Class distribution graph rendered.", "ok")

    # ── View Log toggle ────────────────────────────────────────────────────
    def _toggle_log(self) -> None:
        if self._log_visible:
            self._log_frame.pack_forget()
            self.view_log_btn.configure(text="☰  View Log")
            self._log_visible = False
        else:
            self._log_frame.pack(fill="both", expand=False, padx=14, pady=(6, 10))
            self.view_log_btn.configure(text="☰  Hide Log")
            self._log_visible = True

    def _auto_fix(self):

        result = execute_auto_fixes(
            self.dataset_path,
            self.issue_details,
            fix_invalid=True,
            delete_invalid_images=True,
            log_callback=self._log_safe
        )

        messagebox.showinfo(
            "Auto Fix",
            "Auto Fix completed.\n\n"
            f"Invalid labels removed : {result['invalid_labels']}\n"
            f"Images removed         : {result['invalid_images']}"
    )

        self._start_verification()

    # ── View Issues window ─────────────────────────────────────────────────
    def _open_issues_window(self) -> None:
        win = tk.Toplevel(self)
        win.title("Issue Details")
        win.geometry("720x540")
        win.minsize(500, 400)
        win.configure(bg=BG)
        tk.Label(win, text="Issue Details",
                 font=("Segoe UI", 13, "bold"), bg=BG, fg=ACCENT2
                 ).pack(padx=16, pady=(14, 6), anchor="w")
        frame = tk.Frame(win, bg=BG)
        frame.pack(fill="both", expand=True, padx=14, pady=4)
        text = tk.Text(frame, wrap="word", bg=PANEL, fg=TEXT,
                       font=("Consolas", 10), relief="flat",
                       padx=10, pady=8, insertbackground=TEXT)
        scroll = ttk.Scrollbar(frame, command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        text.tag_configure("cat",  foreground=ACCENT, font=("Consolas", 10, "bold"))
        text.tag_configure("ok",   foreground=SUCCESS)
        text.tag_configure("item", foreground=DANGER)
        for category, files in self.issue_details.items():
            text.insert(tk.END, f"=== {category} ===\n", "cat")
            if files:
                for f in files:
                    text.insert(tk.END, f"  {f}\n", "item")
            else:
                text.insert(tk.END, "  None\n", "ok")
            text.insert(tk.END, "\n")
        text.configure(state="disabled")
        ttk.Button(win, text="Close", command=win.destroy).pack(pady=10)


if __name__ == "__main__":
    app = DatasetQATool()
    app.mainloop()


    