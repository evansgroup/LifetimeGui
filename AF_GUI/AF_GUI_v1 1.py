# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 16:29:49 2025

@author: foun7
"""

"""
Created on Thu Jan 16 14:57:24 2025

@author: Wonsang Hwang
"""
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import base64
from io import BytesIO
import numpy as np
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


matplotlib.use("Agg")

############################################################################
# 1) Helper Functions
############################################################################
def array_to_tk_image(arr, new_size=(400, 400)):
    """
    Convert a 2D NumPy array into a Tkinter PhotoImage (grayscale).
    Also returns the underlying PIL image so we can save it if needed.
    """
    if arr.size == 0:
        # Fallback for empty arrays
        blank_pil = Image.new('L', new_size, color=128)
        return ImageTk.PhotoImage(blank_pil), blank_pil

    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min < 1e-12:
        arr_norm = np.zeros_like(arr, dtype=np.uint8)
    else:
        arr_norm = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)

    pil_img = Image.fromarray(arr_norm, mode='L')
    pil_img = pil_img.resize(new_size, Image.NEAREST)
    return ImageTk.PhotoImage(pil_img), pil_img


def Median_filter(img, n_conv=3):
    """
    Simple median-like filter with a custom kernel.
    """
    if img.ndim != 2:
        return img
    fil = np.ones((n_conv, n_conv))
    mid = n_conv // 2
    fil[mid, mid] = n_conv - 1
    convolved = signal.convolve2d(img, fil, boundary='symm', mode='same')
    return convolved / fil.sum()


############################################################################
# 2) Base64-encoded single combined logo
############################################################################
LOGO_COMBINED_B64 = """
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
"""

############################################################################
# 3) Main GUI Class
############################################################################
class PhasorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Phasor Analysis GUI (Larger images, flexible layout)")
        self.geometry("1600x900")

        # Graceful close
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Main container
        self.main_frame = tk.Frame(self)
        self.main_frame.pack(fill="both", expand=True)

        # Row config: top=0, bottom=1
        self.main_frame.rowconfigure(0, weight=0)
        self.main_frame.rowconfigure(1, weight=1)
        self.main_frame.columnconfigure(0, weight=1)

        # Variables for file paths
        self.inten_file = tk.StringVar()
        self.g_file = tk.StringVar()
        self.s_file = tk.StringVar()

        # Numeric parameters
        self.freq_var = tk.DoubleVar(value=50.0)
        self.n_av_var = tk.IntVar(value=3)

        # Scenario / reference phasors
        self.scenario_var = tk.StringVar(value="Select method")
        self.zr_var_g = tk.DoubleVar(value=0.48)
        self.zr_var_s = tk.DoubleVar(value=0.45)
        self.za_var_g = tk.DoubleVar(value=0.8)
        self.za_var_s = tk.DoubleVar(value=0.3)

        # Data arrays
        self.raw_img_conf = None
        self.raw_img_G    = None
        self.raw_img_S    = None

        # Arrays to be saved as .npy
        self.phasor_array = None  # 3D RGBA data
        self.if_array = None      # 2D float array for IF
        self.af_array = None      # 2D float array for AF

        # PhotoImages
        self.photo_inten  = None
        self.photo_g      = None
        self.photo_s      = None
        self.photo_phasor = None
        self.photo_if     = None
        self.photo_af     = None

        # Build the top panel (inputs + logo) and bottom panel (images + save)
        self.create_top_panel()
        self.create_image_panel()

    ########################################################################
    # On closing
    ########################################################################
    def on_closing(self):
        self.destroy()

    ########################################################################
    # base64 -> PhotoImage
    ########################################################################
    def base64_to_photoimage(self, b64_str, w=200, h=80):
        try:
            bdata = base64.b64decode(b64_str)
            bio = BytesIO(bdata)
            pil_img = Image.open(bio)
            pil_img.load()
            bio.close()

            pil_img = pil_img.resize((w, h), Image.LANCZOS)
            return ImageTk.PhotoImage(pil_img)
        except Exception as e:
            print("Logo decode error:", e)
            return None

    ########################################################################
    # 3.1) Top panel
    ########################################################################
    def create_top_panel(self):
        top_container = tk.Frame(self.main_frame)
        top_container.grid(row=0, column=0, sticky="ew")

        # Left: inputs
        left_frame = tk.Frame(top_container)
        left_frame.pack(side="left", anchor="nw", padx=5, pady=5)

        row_idx = 0
        tk.Label(left_frame, text="Intensity file (.npy):",
                 font=("Arial", 14, "bold")).grid(row=row_idx, column=0, sticky="e")
        tk.Entry(left_frame, textvariable=self.inten_file, width=55,
                 font=("Arial", 14)).grid(row=row_idx, column=1, padx=5)
        tk.Button(left_frame, text="Browse", command=self.browse_intensity,
                  font=("Arial", 14, "bold")).grid(row=row_idx, column=2, padx=5)

        row_idx += 1
        tk.Label(left_frame, text="G file (.npy):",
                 font=("Arial", 14, "bold")).grid(row=row_idx, column=0, sticky="e")
        tk.Entry(left_frame, textvariable=self.g_file, width=55,
                 font=("Arial", 14)).grid(row=row_idx, column=1, padx=5)
        tk.Button(left_frame, text="Browse", command=self.browse_g,
                  font=("Arial", 14, "bold")).grid(row=row_idx, column=2, padx=5)

        row_idx += 1
        tk.Label(left_frame, text="S file (.npy):",
                 font=("Arial", 14, "bold")).grid(row=row_idx, column=0, sticky="e")
        tk.Entry(left_frame, textvariable=self.s_file, width=55,
                 font=("Arial", 14)).grid(row=row_idx, column=1, padx=5)
        tk.Button(left_frame, text="Browse", command=self.browse_s,
                  font=("Arial", 14, "bold")).grid(row=row_idx, column=2, padx=5)

        row_idx += 1
        tk.Label(left_frame, text="Modulation freq (MHz):",
                 font=("Arial", 14, "bold")).grid(row=row_idx, column=0, sticky="e")
        tk.Entry(left_frame, textvariable=self.freq_var, width=10,
                 font=("Arial", 14)).grid(row=row_idx, column=1, sticky="w")

        row_idx += 1
        tk.Label(left_frame, text="Median filter (kernel size):",
                 font=("Arial", 14, "bold")).grid(row=row_idx, column=0, sticky="e")
        tk.Entry(left_frame, textvariable=self.n_av_var, width=10,
                 font=("Arial", 14)).grid(row=row_idx, column=1, sticky="w")

        row_idx += 1
        tk.Label(left_frame, text="Reference phasor (IF):",
                 font=("Arial", 14, "bold")).grid(row=row_idx, column=0, sticky="e")
        if_frame = tk.Frame(left_frame)
        if_frame.grid(row=row_idx, column=1, sticky="w")
        tk.Label(if_frame, text="G", font=("Arial", 14, "bold")).pack(side="left")
        tk.Entry(if_frame, textvariable=self.zr_var_g, width=5,
                 font=("Arial", 14)).pack(side="left", padx=(2, 10))
        tk.Label(if_frame, text="S", font=("Arial", 14, "bold")).pack(side="left")
        tk.Entry(if_frame, textvariable=self.zr_var_s, width=5,
                 font=("Arial", 14)).pack(side="left", padx=(2, 0))

        row_idx += 1
        tk.Label(left_frame, text="Reference phasor (AF):",
                 font=("Arial", 14, "bold")).grid(row=row_idx, column=0, sticky="e")
        af_frame = tk.Frame(left_frame)
        af_frame.grid(row=row_idx, column=1, sticky="w")
        tk.Label(af_frame, text="G", font=("Arial", 14, "bold")).pack(side="left")
        tk.Entry(af_frame, textvariable=self.za_var_g, width=5,
                 font=("Arial", 14)).pack(side="left", padx=(2, 10))
        tk.Label(af_frame, text="S", font=("Arial", 14, "bold")).pack(side="left")
        tk.Entry(af_frame, textvariable=self.za_var_s, width=5,
                 font=("Arial", 14)).pack(side="left", padx=(2, 0))

        row_idx += 1
        tk.Label(left_frame, text="Select scenario:",
                 font=("Arial", 14, "bold")).grid(row=row_idx, column=0, sticky="e")
        scenarios = ["Both references known", "Only IF known"]
        tk.OptionMenu(left_frame, self.scenario_var, *scenarios).grid(
            row=row_idx, column=1, sticky="w", padx=5)

        row_idx += 1
        tk.Button(left_frame, text="Compute", command=self.compute_results,
                  font=("Arial", 14, "bold")).grid(row=row_idx, column=1, sticky="w", pady=5)

        # Right: single combined logo
        right_frame = tk.Frame(top_container)
        right_frame.pack(side="right", anchor="ne", padx=5, pady=5)

        combined_logo_tk = self.base64_to_photoimage(LOGO_COMBINED_B64, w=300, h=100)
        if combined_logo_tk:
            tk.Label(right_frame, image=combined_logo_tk).pack()
            self.combined_logo_ref = combined_logo_tk
        else:
            tk.Label(right_frame, text="(Logos here)", font=("Arial", 14, "bold")).pack(pady=10)

        tk.Label(right_frame, text="© 2025 Evans Lab. All rights reserved.",
                 font=("Arial", 12, "bold")).pack(pady=10)

    ########################################################################
    # 3.2) Bottom panel for images + save buttons
    ########################################################################
    def create_image_panel(self):
        self.image_frame = tk.Frame(self.main_frame)
        self.image_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        for r in range(2):
            self.image_frame.rowconfigure(r, weight=1)
        for c in range(3):
            self.image_frame.columnconfigure(c, weight=1)

        # Row 0: Intensity, G, S
        self.label_inten = tk.Label(self.image_frame, text="Intensity",
                                    bg="gray", font=("Arial", 18, "bold"))
        self.label_inten.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.label_g = tk.Label(self.image_frame, text="G",
                                bg="gray", font=("Arial", 18, "bold"))
        self.label_g.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.label_s = tk.Label(self.image_frame, text="S",
                                bg="gray", font=("Arial", 18, "bold"))
        self.label_s.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

        # Row 1: Phasor, IF, AF
        self.label_lifetime = tk.Label(self.image_frame, text="Phasor",
                                       bg="gray", font=("Arial", 18, "bold"))
        self.label_lifetime.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        self.label_if = tk.Label(self.image_frame, text="IF",
                                 bg="gray", font=("Arial", 18, "bold"))
        self.label_if.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        self.label_af = tk.Label(self.image_frame, text="AF",
                                 bg="gray", font=("Arial", 18, "bold"))
        self.label_af.grid(row=1, column=2, padx=5, pady=5, sticky="nsew")

        # Save buttons (same font as 'Compute')
        button_frame = tk.Frame(self.image_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=5)


        tk.Button(button_frame, text="Save IF", command=self.save_if,
                  font=("Arial", 14, "bold")).pack(side="left", padx=10)
        tk.Button(button_frame, text="Save AF", command=self.save_af,
                  font=("Arial", 14, "bold")).pack(side="left", padx=10)

    ########################################################################
    # File dialogs
    ########################################################################
    def browse_intensity(self):
        fpath = filedialog.askopenfilename(filetypes=[("NumPy", "*.npy")])
        if fpath:
            self.inten_file.set(fpath)
            self.try_load_images()

    def browse_g(self):
        fpath = filedialog.askopenfilename(filetypes=[("NumPy", "*.npy")])
        if fpath:
            self.g_file.set(fpath)
            self.try_load_images()

    def browse_s(self):
        fpath = filedialog.askopenfilename(filetypes=[("NumPy", "*.npy")])
        if fpath:
            self.s_file.set(fpath)
            self.try_load_images()

    def try_load_images(self):
        if self.inten_file.get() and self.g_file.get() and self.s_file.get():
            self.load_images()

    ########################################################################
    # Load images (.npy) => top row
    ########################################################################
    def load_images(self):
        try:
            conf_path = self.inten_file.get()
            g_path    = self.g_file.get()
            s_path    = self.s_file.get()

            if not (conf_path and g_path and s_path):
                raise ValueError("Please select all three .npy files.")

            self.raw_img_conf = np.load(conf_path)
            self.raw_img_G    = np.load(g_path)
            self.raw_img_S    = np.load(s_path)
        except Exception as e:
            messagebox.showerror("Load error", str(e))
            return

        # Basic cleaning/clamping
        self.raw_img_G[self.raw_img_G > 1.5] = 1.5
        self.raw_img_G[self.raw_img_G < 0]   = 0
        self.raw_img_S[self.raw_img_S > 1]   = 1
        self.raw_img_S[self.raw_img_S < 0]   = 0
        self.raw_img_conf[np.isnan(self.raw_img_conf)] = 0
        self.raw_img_G[np.isnan(self.raw_img_G)]       = 0
        self.raw_img_S[np.isnan(self.raw_img_S)]       = 0

        # Convert arrays to PhotoImages
        self.photo_inten, _ = array_to_tk_image(self.raw_img_conf, (400, 400))
        self.photo_g,     _ = array_to_tk_image(self.raw_img_G,    (400, 400))
        self.photo_s,     _ = array_to_tk_image(self.raw_img_S,    (400, 400))

        self.label_inten.config(image=self.photo_inten, text="")
        self.label_inten.image = self.photo_inten

        self.label_g.config(image=self.photo_g, text="")
        self.label_g.image = self.photo_g

        self.label_s.config(image=self.photo_s, text="")
        self.label_s.image = self.photo_s

    ########################################################################
    # Compute => bottom row
    ########################################################################
    def compute_results(self):
        if (self.raw_img_conf is None
                or self.raw_img_G is None
                or self.raw_img_S is None):
            messagebox.showerror("Compute error", "Images not loaded yet.")
            return

        # 1) Filter
        conf_f = Median_filter(self.raw_img_conf, self.n_av_var.get())
        g_f    = Median_filter(self.raw_img_G,    self.n_av_var.get())
        s_f    = Median_filter(self.raw_img_S,    self.n_av_var.get())

        # 2) Scenario logic
        scenario = self.scenario_var.get()
        z_r = [self.zr_var_g.get(), self.zr_var_s.get()]
        z_a = [self.za_var_g.get(), self.za_var_s.get()]
        d = np.sqrt((g_f - z_r[0])**2 + (s_f - z_r[1])**2)
        dis_maj = 2 * np.std(d[conf_f > np.mean(conf_f)])
        if scenario == "Both references known":
            dist_if = np.sqrt((g_f - z_r[0])**2 + (s_f - z_r[1])**2)
            dist_af = np.sqrt((z_a[0] - z_r[0])**2 + (z_a[1] - z_r[1])**2)
            frac_af = dist_if / (dist_af + 1e-12)
            frac_if = 1 - frac_af
            frac_if[frac_if < 0] = 0
            frac_if[frac_if > 1] = 1
        elif scenario == "Only IF known":
            dist_af = np.sqrt((g_f - z_r[0])**2 + (s_f - z_r[1])**2)
            dist_af[dist_af > dis_maj] = dis_maj
            frac_af = dist_af / (dis_maj + 1e-12)
            frac_if = 1 - frac_af
            frac_if[frac_if < 0] = 0
            frac_if[frac_if > 1] = 1
        else:
            # No references, or anything else
            dist_af = np.sqrt((g_f - z_r[0])**2 + (s_f - z_r[1])**2)
            dist_af[dist_af > dis_maj] = dis_maj
            frac_af = dist_af / (dis_maj + 1e-12)
            frac_if = 1 - frac_af
            frac_if[frac_if < 0] = 0
            frac_if[frac_if > 1] = 1
            
        # Build IF/AF arrays
        IF_img = frac_if * conf_f
        AF_img = frac_af * conf_f

        # Store them for saving
        self.if_array = IF_img
        self.af_array = AF_img

        # 3) Phasor figure in memory
        f_mhz = self.freq_var.get()
        w = 2.0 * np.pi * (f_mhz * 1e6)

        fig_phasor = plt.figure(figsize=(4, 4), dpi=100)
        ax = fig_phasor.add_subplot(111)

        x = np.arange(0.0, 1.1, 0.001)
        circle_raw = 0.25 - (x - 0.5)**2
        circle_clipped = np.clip(circle_raw, 0, None)
        circle = np.sqrt(circle_clipped)
        ax.plot(x, circle, 'b')

        # Add diagonal lines and annotate with time points
        for k_ in range(1, 6):  # Iterate for 1ns to 5ns
            denom = np.real(1 / (1 - 1j * (w / 1e9) * k_))
            if denom != 0:
                y_ = (np.imag(1 / (1 - 1j * (w / 1e9) * k_)) / denom) * x
                mask = x < denom
                ax.plot(x[mask], y_[mask], 'b')
                # Annotate near the diagonal line
                label_x = denom / 1.05 if denom > 0.8 else denom * 1.05
                label_y = y_[mask][-1] * 1.05
                ax.text(label_x, label_y, f'{k_} ns', color='blue', fontsize=8)
        
        # Add the histogram
        mask_int = (conf_f > 100)
        ax.hist2d(
            g_f[mask_int],
            s_f[mask_int],
            bins=300,
            range=[[-0.05, 1.1], [0.0, 0.6]],
            cmap='jet',
            cmin=20,
            norm=LogNorm()
        )
        
        # Add the reference marker 'X'
        ax.scatter(z_r[0], z_r[1], c='black', marker='X', label='Reference Phasor (IF)')
        ax.legend(loc='upper right', fontsize=8)
        
        # Customize axes
        ax.set_xlim([0, 1.1])
        ax.set_ylim([0, 0.6])
        ax.tick_params(left=False, right=False, labelleft=False,
                       labelbottom=False, bottom=False)


        # Render to RGBA buffer
        fig_phasor.canvas.draw()
        width, height = fig_phasor.canvas.get_width_height()
        img_data = np.frombuffer(fig_phasor.canvas.buffer_rgba(), dtype=np.uint8)
        img_data = img_data.reshape((height, width, 4))  # RGBA
        plt.close(fig_phasor)

        # Keep the array for saving
        self.phasor_array = img_data

        # Convert to PhotoImage for display
        pil_rgba = Image.fromarray(img_data, mode='RGBA').resize((400, 400), Image.LANCZOS)
        self.photo_phasor = ImageTk.PhotoImage(pil_rgba)

        self.label_lifetime.config(image=self.photo_phasor, text="")
        self.label_lifetime.image = self.photo_phasor

        # IF/AF as PhotoImages
        self.photo_if, _ = array_to_tk_image(IF_img, (400, 400))
        self.label_if.config(image=self.photo_if, text="")
        self.label_if.image = self.photo_if

        self.photo_af, _ = array_to_tk_image(AF_img, (400, 400))
        self.label_af.config(image=self.photo_af, text="")
        self.label_af.image = self.photo_af

        messagebox.showinfo("Compute", "Computation complete.")

    ########################################################################
    # Save IF and AF => .npy
    ########################################################################


    def save_if(self):
        """
        Save the IF array as a .npy file.
        """
        if self.if_array is None:
            messagebox.showerror("Save error", "No IF array to save. Please Compute first.")
            return
    
        fpath = filedialog.asksaveasfilename(defaultextension=".npy")
        if fpath:
            # Save the NumPy array (2D float) to .npy
            np.save(fpath, self.if_array)
            messagebox.showinfo("Saved", f"IF array saved to:\n{fpath}")
    
    def save_af(self):
        """
        Save the AF array as a .npy file.
        """
        if self.af_array is None:
            messagebox.showerror("Save error", "No AF array to save. Please Compute first.")
            return
    
        fpath = filedialog.asksaveasfilename(defaultextension=".npy")
        if fpath:
            # Save the NumPy array (2D float) to .npy
            np.save(fpath, self.af_array)
            messagebox.showinfo("Saved", f"AF array saved to:\n{fpath}")


############################################################################
# Run
############################################################################
if __name__ == "__main__":
    app = PhasorGUI()
    app.mainloop()