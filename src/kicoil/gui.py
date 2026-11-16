# Copyright 2025 Jan Sebastian Götte <code@jaseg.de>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file was created using claude.ai because hand-writing GUI code sucks. The rest of kicoil is written by hand.
#

""" GUI for generating KiCad footprints using kicoil. """

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import sys
import logging
import multiprocessing
import traceback
from pathlib import Path
from functools import partial
from io import BytesIO

from .geometry import divisors, CircleShape, SectorShape, StarShape, SVGShape, RectangleShape, RegularPolygonShape, TrapezoidShape, DEFAULT_CIRCLE_SEGMENTS
from .svg import make_transparent_svg
from .worker import build_model
from .kicad import update_footprint_instance

from kipy.board_types import FootprintInstance, BoardLayer

try:
    # for rendering gerbonara's svg output to PNG
    import cairosvg
    # for scaling the rendered PNGs to the UI's resolution since tkinter is very limited there
    from PIL import Image, ImageTk
    HAS_PREVIEW = True
except ImportError:
    HAS_PREVIEW = False

NEW_FOOTPRINT_REFERENCE = 'L?'


class TextWidgetHandler(logging.Handler):
    """Custom logging handler that writes to a tkinter Text widget"""
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
        self.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        
    def emit(self, record):
        msg = self.format(record)

        # Determine tag based on log level
        if record.levelno >= logging.ERROR:
            tag = 'error'
        elif record.levelno >= logging.WARNING:
            tag = 'warning'
        else:
            tag = 'info'

        # Temporarily enable widget to insert text
        self.text_widget['state'] = 'normal'
        self.text_widget.insert(tk.END, msg + '\n', tag)
        self.text_widget.see(tk.END)
        self.text_widget['state'] = 'disabled'
        self.text_widget.update_idletasks()


# https://stackoverflow.com/questions/27820178/how-to-add-placeholder-to-an-entry-in-tkinter
class EntryWithPlaceholder(tk.Entry):
    def __init__(self, master=None, placeholder="empty", color='grey', *args, **kwargs):
        # Initialize parent Entry with all provided arguments
        super().__init__(master, *args, **kwargs)

        self.placeholder = placeholder
        self.placeholder_color = color
        self.showing_placeholder = False

        # Get the default foreground color from the created widget
        self.default_fg_color = self['fg']

        self.bind("<FocusIn>", self.foc_in)
        self.bind("<FocusOut>", self.foc_out)

        # Check initial state and show placeholder if empty
        # Use a small delay to ensure the widget is fully initialized
        self.after(1, self._check_initial_state)

    def _check_initial_state(self):
        if not super().get():
            self.put_placeholder()

    def put_placeholder(self):
        self.showing_placeholder = True
        self['fg'] = self.placeholder_color
        super().delete(0, 'end')
        super().insert(0, self.placeholder)

    def update_placeholder(self, new_placeholder):
        self.placeholder = new_placeholder
        # If currently showing placeholder, update the display
        if self.showing_placeholder:
            super().delete(0, 'end')
            super().insert(0, self.placeholder)

    def foc_in(self, *args):
        if self.showing_placeholder:
            self.showing_placeholder = False
            self['fg'] = self.default_fg_color
            super().delete(0, 'end')

    def foc_out(self, *args):
        if not super().get():
            self.put_placeholder()
        else:
            self.showing_placeholder = False

    def get(self):
        if self.showing_placeholder:
            return ''
        return super().get()


class KiCoilGUI:
    def __init__(self, root, kicad_inst=None):
        self.kicad_inst = kicad_inst
        self.root = root
        self.root.title("KiCoil - Planar Inductor Generator")
        self.root.geometry("1000x650")

        # Register validation command for non-negative numbers early
        self.validate_nonneg_cmd = self.root.register(self.validate_nonnegative)

        style = ttk.Style()
        style.theme_use('clam')

        main_container = ttk.Frame(root)
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_container.columnconfigure(0, weight=0)  # Left panel doesn't grow horizontally
        main_container.columnconfigure(1, weight=1)  # Preview panel grows to fill space
        main_container.rowconfigure(0, weight=1)

        main_frame = ttk.Frame(main_container, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        ttk.Label(title_frame, text="Planar Inductor Generator",
                 font=('Helvetica', 16, 'bold')).pack(side=tk.LEFT)

        self.preview_visible = tk.BooleanVar(value=True)
        self.preview_button = ttk.Button(title_frame, text="Hide Preview",
                  command=self.toggle_preview, width=15)
        self.preview_button.pack(side=tk.RIGHT)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        geometry_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(geometry_frame, text="Geometry")
        self.create_geometry_params(geometry_frame)

        # Shape parameter tabs (will be shown/hidden based on selection)
        self.shape_param_frames = {}

        circle_frame = ttk.Frame(self.notebook, padding="10")
        self.shape_param_frames["Circle"] = circle_frame
        self.notebook.add(circle_frame, text="Circle Parameters")
        self.create_circle_params(circle_frame)

        rectangle_frame = ttk.Frame(self.notebook, padding="10")
        self.shape_param_frames["Rectangle"] = rectangle_frame
        self.notebook.add(rectangle_frame, text="Rectangle Parameters")
        self.create_rectangle_params(rectangle_frame)

        trapezoid_frame = ttk.Frame(self.notebook, padding="10")
        self.shape_param_frames["Trapezoid"] = trapezoid_frame
        self.notebook.add(trapezoid_frame, text="Trapezoid Parameters")
        self.create_trapezoid_params(trapezoid_frame)

        sector_frame = ttk.Frame(self.notebook, padding="10")
        self.shape_param_frames["Sector"] = sector_frame
        self.notebook.add(sector_frame, text="Sector Parameters")
        self.create_sector_params(sector_frame)

        star_frame = ttk.Frame(self.notebook, padding="10")
        self.shape_param_frames["Star"] = star_frame
        self.notebook.add(star_frame, text="Star Parameters")
        self.create_star_params(star_frame)

        polygon_frame = ttk.Frame(self.notebook, padding="10")
        self.shape_param_frames["Regular Polygon"] = polygon_frame
        self.notebook.add(polygon_frame, text="Polygon Parameters")
        self.create_polygon_params(polygon_frame)

        svg_frame = ttk.Frame(self.notebook, padding="10")
        self.shape_param_frames["SVG"] = svg_frame
        self.notebook.add(svg_frame, text="SVG Parameters")
        self.create_svg_params(svg_frame)

        traces_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(traces_frame, text="Traces")
        self.create_trace_params(traces_frame)

        via_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(via_frame, text="Vias")
        self.create_via_params(via_frame)

        output_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(output_frame, text="Output")
        self.create_output_params(output_frame)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        ttk.Button(button_frame, text="Show Valid Twists", 
                  command=self.show_valid_twists, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Footprint File", 
                  command=self.save_footprint_file, width=20).pack(side=tk.LEFT, padx=5)
        self.cancel_button = ttk.Button(button_frame, text="Cancel", command=self.cancel_layout, width=10,
                                        state='disabled')
        self.cancel_button.pack(side=tk.LEFT, padx=5)
        self.status_var = tk.StringVar()
        ttk.Label(button_frame, textvariable=self.status_var, foreground='gray').pack(side=tk.LEFT, padx=5)
        if self.kicad_inst:
            ttk.Button(button_frame, text="Place on Board",
                      command=self.update_board_footprint, width=20).pack(side=tk.LEFT, padx=5)
        
        status_label = ttk.Label(main_frame, text="Output:", font=('Helvetica', 10, 'bold'))
        status_label.grid(row=3, column=0, sticky=tk.W, pady=(10, 0))
        
        self.output_text = scrolledtext.ScrolledText(main_frame, height=8, width=80, state='disabled')
        self.output_text.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        self.output_text.tag_config('error', foreground='red')
        self.output_text.tag_config('warning', foreground='orange')
        self.output_text.tag_config('info', foreground='black')

        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=0)  # Notebook doesn't grow vertically
        main_frame.rowconfigure(4, weight=1)  # Output area takes all vertical space

        self.preview_frame = ttk.LabelFrame(main_container, text="Preview", padding=10)

        if HAS_PREVIEW:
            # Create canvas for image display
            self.preview_canvas = tk.Canvas(self.preview_frame, bg='white')
            self.preview_canvas.pack(fill=tk.BOTH, expand=True)
            self.preview_image = None  # Store reference to prevent garbage collection
            self.preview_raw_image = None  # Store unscaled image for rescaling

            # Bind canvas resize event to update preview
            self.preview_canvas.bind('<Configure>', self._on_preview_resize)
        
        else:
            info_text = "Preview not available\n\nInstall dependencies:\npip install cairosvg pillow"
            self.preview_label = ttk.Label(self.preview_frame, text=info_text,
                                           justify=tk.CENTER, anchor=tk.CENTER)
            self.preview_label.pack(fill=tk.BOTH, expand=True)

        self.preview_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W), padx=(10, 10), pady=10)

        self.current_model = None
        self._validation_after_id = None
        self.layout_process = None

        self.setup_logging()
        self.setup_traces()
        self.update_shape_tab_visibility()  # Initialize tab visibility
        self.root.after(100, self.start_layout)

    def validate_nonnegative(self, value_if_allowed):
        """Validation callback for spinboxes to prevent negative values"""
        if value_if_allowed == "" or value_if_allowed == "-":
            # Allow empty string (user is typing) but not standalone minus
            return value_if_allowed == ""
        try:
            float_val = float(value_if_allowed)
            return float_val >= 0
        except ValueError:
            return False

    def _on_preview_resize(self, event):
        # Debounce resize events - only update after resize is complete
        if hasattr(self, '_resize_after_id') and self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(100, self._rescale_preview)

    def _rescale_preview(self):
        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        if canvas_width <= 1 or canvas_height <= 1 or self.preview_raw_image is None:
            self.preview_canvas.delete("all")
            return

        # Calculate scaling to fit within canvas
        image = self.preview_raw_image.copy()
        img_ratio = image.width / image.height
        canvas_ratio = canvas_width / canvas_height

        if img_ratio > canvas_ratio:
            # Image is wider than canvas
            new_width = int(canvas_width * 0.95)
            new_height = int(new_width / img_ratio)
        else:
            # Image is taller than canvas
            new_height = int(canvas_height * 0.95)
            new_width = int(new_height * img_ratio)

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        self.preview_image = ImageTk.PhotoImage(image)
        self.preview_canvas.create_image(
            canvas_width // 2, canvas_height // 2,
            image=self.preview_image, anchor=tk.CENTER
        )

    def toggle_preview(self):
        if self.preview_visible.get():
            self.preview_frame.grid_forget()
            self.preview_visible.set(False)
            self.preview_button.config(text="Show Preview")

            current_width = self.root.winfo_width()
            current_height = self.root.winfo_height()
            new_width = max(680, current_width - 400)
            self.root.geometry(f"{new_width}x{current_height}")

        else:
            self.preview_frame.grid(row=0, column=1, sticky=(tk.N, tk.S, tk.E, tk.W), padx=(10, 0), pady=10)
            self.preview_visible.set(True)
            self.preview_button.config(text="Hide Preview")

            current_width = self.root.winfo_width()
            current_height = self.root.winfo_height()
            new_width = current_width + 400  # Add ~400px for preview
            self.root.geometry(f"{new_width}x{current_height}")

    def create_geometry_params(self, parent):
        row = 0

        # Shape Type
        ttk.Label(parent, text="Shape Type:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.shape_type_var = tk.StringVar(value="Circle")
        shape_combo = ttk.Combobox(parent, textvariable=self.shape_type_var,
                                   values=["Circle", "Rectangle", "Trapezoid", "Sector", "Star", "Regular Polygon", "SVG"],
                                   state='readonly', width=23)
        shape_combo.grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Label(parent, text="Coil outline shape",
                 foreground='gray').grid(row=row, column=2, sticky=tk.W, padx=(10, 0))
        row += 1

        # Turns
        ttk.Label(parent, text="Number of Turns:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.turns_var = tk.IntVar(value=7)
        ttk.Spinbox(parent, from_=1, to=100, textvariable=self.turns_var,
                   width=15,
                   validate='key', validatecommand=(self.validate_nonneg_cmd, '%P')).grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Label(parent, text="Number of spiral turns",
                 foreground='gray').grid(row=row, column=2, sticky=tk.W, padx=(10, 0))
        row += 1

        # Twists
        ttk.Label(parent, text="Twists per Revolution:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.twists_var = tk.IntVar(value=4)
        ttk.Spinbox(parent, from_=0, to=50, textvariable=self.twists_var,
                   width=15,
                   validate='key', validatecommand=(self.validate_nonneg_cmd, '%P')).grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Label(parent, text="Must be co-prime to turns",
                 foreground='gray').grid(row=row, column=2, sticky=tk.W, padx=(10, 0))
        row += 1

        # Layer Mode
        ttk.Label(parent, text="Layer Mode:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.layer_mode_var = tk.IntVar(value=2)
        layer_frame = ttk.Frame(parent)
        layer_frame.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=5)
        ttk.Radiobutton(layer_frame, text="Two Layer", variable=self.layer_mode_var,
                       value=2).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(layer_frame, text="Single Layer", variable=self.layer_mode_var,
                       value=1).pack(side=tk.LEFT)
        row += 1

        # Direction
        ttk.Label(parent, text="Winding Direction:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.direction_var = tk.StringVar(value="counter-clockwise")
        dir_frame = ttk.Frame(parent)
        dir_frame.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=5)
        ttk.Radiobutton(dir_frame, text="Counter-Clockwise", variable=self.direction_var,
                       value="counter-clockwise").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(dir_frame, text="Clockwise", variable=self.direction_var,
                       value="clockwise").pack(side=tk.LEFT)
        row += 1

    def create_circle_params(self, parent):
        row = 0
        ttk.Label(parent, text="Outer Diameter (mm):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.circle_outer_dia_var = tk.DoubleVar(value=50.0)
        ttk.Spinbox(parent, from_=1, to=500, increment=0.5,
                   textvariable=self.circle_outer_dia_var, width=15,
                   validate='key', validatecommand=(self.validate_nonneg_cmd, '%P')).grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Label(parent, text="Outside diameter of coil",
                 foreground='gray').grid(row=row, column=2, sticky=tk.W, padx=(10, 0))
        row += 1

        ttk.Label(parent, text="Inner Diameter (mm):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.circle_inner_dia_var = tk.DoubleVar(value=25.0)
        ttk.Spinbox(parent, from_=0, to=500, increment=0.5,
                   textvariable=self.circle_inner_dia_var, width=15,
                   validate='key', validatecommand=(self.validate_nonneg_cmd, '%P')).grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Label(parent, text="Inside diameter of coil",
                 foreground='gray').grid(row=row, column=2, sticky=tk.W, padx=(10, 0))
        row += 1

    def create_rectangle_params(self, parent):
        row = 0
        ttk.Label(parent, text="Width (mm):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.rect_width_var = tk.DoubleVar(value=50.0)
        ttk.Spinbox(parent, from_=1, to=500, increment=0.5,
                   textvariable=self.rect_width_var, width=15,
                   validate='key', validatecommand=(self.validate_nonneg_cmd, '%P')).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        ttk.Label(parent, text="Height (mm):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.rect_height_var = tk.DoubleVar(value=40.0)
        ttk.Spinbox(parent, from_=1, to=500, increment=0.5,
                   textvariable=self.rect_height_var, width=15,
                   validate='key', validatecommand=(self.validate_nonneg_cmd, '%P')).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        ttk.Label(parent, text="Annular Width (mm):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.rect_annular_width_var = tk.DoubleVar(value=10.0)
        ttk.Spinbox(parent, from_=1, to=100, increment=0.5,
                   textvariable=self.rect_annular_width_var, width=15,
                   validate='key', validatecommand=(self.validate_nonneg_cmd, '%P')).grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Label(parent, text="Width of trace area",
                 foreground='gray').grid(row=row, column=2, sticky=tk.W, padx=(10, 0))
        row += 1

    def create_trapezoid_params(self, parent):
        row = 0
        ttk.Label(parent, text="Width (mm):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.trap_width_var = tk.DoubleVar(value=50.0)
        ttk.Spinbox(parent, from_=1, to=500, increment=0.5,
                   textvariable=self.trap_width_var, width=15,
                   validate='key', validatecommand=(self.validate_nonneg_cmd, '%P')).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        ttk.Label(parent, text="Height (mm):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.trap_height_var = tk.DoubleVar(value=40.0)
        ttk.Spinbox(parent, from_=1, to=500, increment=0.5,
                   textvariable=self.trap_height_var, width=15,
                   validate='key', validatecommand=(self.validate_nonneg_cmd, '%P')).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        ttk.Label(parent, text="Offset (mm):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.trap_offset_var = tk.DoubleVar(value=10.0)
        ttk.Spinbox(parent, from_=0, to=100, increment=0.5,
                   textvariable=self.trap_offset_var, width=15,
                   validate='key', validatecommand=(self.validate_nonneg_cmd, '%P')).grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Label(parent, text="Corner offset at shorter edge",
                 foreground='gray').grid(row=row, column=2, sticky=tk.W, padx=(10, 0))
        row += 1

        ttk.Label(parent, text="Annular Width (mm):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.trap_annular_width_var = tk.DoubleVar(value=10.0)
        ttk.Spinbox(parent, from_=1, to=100, increment=0.5,
                   textvariable=self.trap_annular_width_var, width=15,
                   validate='key', validatecommand=(self.validate_nonneg_cmd, '%P')).grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Label(parent, text="Width of trace area",
                 foreground='gray').grid(row=row, column=2, sticky=tk.W, padx=(10, 0))
        row += 1

    def create_sector_params(self, parent):
        row = 0
        ttk.Label(parent, text="Outer Diameter (mm):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.sector_outer_dia_var = tk.DoubleVar(value=50.0)
        ttk.Spinbox(parent, from_=1, to=500, increment=0.5,
                   textvariable=self.sector_outer_dia_var, width=15,
                   validate='key', validatecommand=(self.validate_nonneg_cmd, '%P')).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        ttk.Label(parent, text="Inner Diameter (mm):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.sector_inner_dia_var = tk.DoubleVar(value=25.0)
        ttk.Spinbox(parent, from_=0, to=500, increment=0.5,
                   textvariable=self.sector_inner_dia_var, width=15,
                   validate='key', validatecommand=(self.validate_nonneg_cmd, '%P')).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        ttk.Label(parent, text="Angle (degrees):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.sector_angle_var = tk.DoubleVar(value=45.0)
        ttk.Spinbox(parent, from_=1, to=360, increment=1.0,
                   textvariable=self.sector_angle_var, width=15,
                   validate='key', validatecommand=(self.validate_nonneg_cmd, '%P')).grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Label(parent, text="Sector angle",
                 foreground='gray').grid(row=row, column=2, sticky=tk.W, padx=(10, 0))
        row += 1

        ttk.Label(parent, text="Annular Width (mm):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.sector_annular_width_var = tk.DoubleVar(value=5.0)
        ttk.Spinbox(parent, from_=1, to=100, increment=0.5,
                   textvariable=self.sector_annular_width_var, width=15,
                   validate='key', validatecommand=(self.validate_nonneg_cmd, '%P')).grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Label(parent, text="Width of trace area",
                 foreground='gray').grid(row=row, column=2, sticky=tk.W, padx=(10, 0))
        row += 1

    def create_star_params(self, parent):
        row = 0
        ttk.Label(parent, text="Outer Diameter (mm):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.star_outer_dia_var = tk.DoubleVar(value=50.0)
        ttk.Spinbox(parent, from_=1, to=500, increment=0.5,
                   textvariable=self.star_outer_dia_var, width=15,
                   validate='key', validatecommand=(self.validate_nonneg_cmd, '%P')).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        ttk.Label(parent, text="Inner Diameter (mm):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.star_inner_dia_var = tk.DoubleVar(value=25.0)
        ttk.Spinbox(parent, from_=0, to=500, increment=0.5,
                   textvariable=self.star_inner_dia_var, width=15,
                   validate='key', validatecommand=(self.validate_nonneg_cmd, '%P')).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        ttk.Label(parent, text="Points:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.star_points_var = tk.IntVar(value=5)
        ttk.Spinbox(parent, from_=3, to=20, textvariable=self.star_points_var,
                   width=15,
                   validate='key', validatecommand=(self.validate_nonneg_cmd, '%P')).grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Label(parent, text="Number of star points",
                 foreground='gray').grid(row=row, column=2, sticky=tk.W, padx=(10, 0))
        row += 1

        ttk.Label(parent, text="Annular Width (mm):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.star_annular_width_var = tk.DoubleVar(value=5.0)
        ttk.Spinbox(parent, from_=1, to=100, increment=0.5,
                   textvariable=self.star_annular_width_var, width=15,
                   validate='key', validatecommand=(self.validate_nonneg_cmd, '%P')).grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Label(parent, text="Width of trace area",
                 foreground='gray').grid(row=row, column=2, sticky=tk.W, padx=(10, 0))
        row += 1

    def create_polygon_params(self, parent):
        row = 0
        ttk.Label(parent, text="Diameter (mm):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.poly_diameter_var = tk.DoubleVar(value=50.0)
        ttk.Spinbox(parent, from_=1, to=500, increment=0.5,
                   textvariable=self.poly_diameter_var, width=15,
                   validate='key', validatecommand=(self.validate_nonneg_cmd, '%P')).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        ttk.Label(parent, text="Corners:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.poly_corners_var = tk.IntVar(value=8)
        ttk.Spinbox(parent, from_=3, to=20, textvariable=self.poly_corners_var,
                   width=15,
                   validate='key', validatecommand=(self.validate_nonneg_cmd, '%P')).grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Label(parent, text="Number of polygon corners",
                 foreground='gray').grid(row=row, column=2, sticky=tk.W, padx=(10, 0))
        row += 1

        ttk.Label(parent, text="Annular Width (mm):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.poly_annular_width_var = tk.DoubleVar(value=10.0)
        ttk.Spinbox(parent, from_=1, to=100, increment=0.5,
                   textvariable=self.poly_annular_width_var, width=15,
                   validate='key', validatecommand=(self.validate_nonneg_cmd, '%P')).grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Label(parent, text="Width of trace area",
                 foreground='gray').grid(row=row, column=2, sticky=tk.W, padx=(10, 0))
        row += 1

    def create_svg_params(self, parent):
        row = 0
        ttk.Label(parent, text="SVG File:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.svg_filename_var = tk.StringVar(value="")
        svg_entry = ttk.Entry(parent, textvariable=self.svg_filename_var, width=30)
        svg_entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(parent, text="Browse...", command=self.browse_svg_file, width=10).grid(row=row, column=2, sticky=tk.W, padx=(5, 0), pady=5)
        row += 1

        ttk.Label(parent, text="Annular Width (mm):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.svg_annular_width_var = tk.DoubleVar(value=5.0)
        ttk.Spinbox(parent, from_=1, to=100, increment=0.5,
                   textvariable=self.svg_annular_width_var, width=15,
                   validate='key', validatecommand=(self.validate_nonneg_cmd, '%P')).grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Label(parent, text="Width of trace area",
                 foreground='gray').grid(row=row, column=2, sticky=tk.W, padx=(10, 0))
        row += 1

    def browse_svg_file(self):
        filename = filedialog.askopenfilename(
            title="Select SVG File",
            filetypes=[("SVG files", "*.svg"), ("All files", "*.*")]
        )
        if filename:
            self.svg_filename_var.set(filename)

    def update_shape_tab_visibility(self, *args):
        """Show only the shape parameter tab for the currently selected shape"""
        selected_shape = self.shape_type_var.get()

        # Hide all shape parameter tabs
        for shape_name, frame in self.shape_param_frames.items():
            tab_id = self.notebook.index(frame)
            self.notebook.tab(tab_id, state='hidden')

        # Show only the selected shape's tab
        if selected_shape in self.shape_param_frames:
            frame = self.shape_param_frames[selected_shape]
            tab_id = self.notebook.index(frame)
            self.notebook.tab(tab_id, state='normal')

    def create_trace_params(self, parent):
        row = 0

        # Trace Width
        ttk.Label(parent, text="Trace Width (mm):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.trace_width_entry = EntryWithPlaceholder(parent, placeholder="automatic", width=15)
        self.trace_width_entry.grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        # Clearance
        ttk.Label(parent, text="Clearance (mm):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.clearance_entry = EntryWithPlaceholder(parent, placeholder="automatic", width=15)
        self.clearance_entry.grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        # Copper Thickness (in µm)
        ttk.Label(parent, text="Copper Thickness (µm):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.copper_thickness_var = tk.DoubleVar(value=35.0)  # 35µm = 0.035mm = 1 Oz
        ttk.Spinbox(parent, from_=1, to=1000, increment=1, format="%.1f",
                   textvariable=self.copper_thickness_var, width=15,
                   validate='key', validatecommand=(self.validate_nonneg_cmd, '%P')).grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Label(parent, text="35µm = 1 Oz copper",
                 foreground='gray').grid(row=row, column=2, sticky=tk.W, padx=(10, 0))
        row += 1
        
    def create_via_params(self, parent):
        """Create via parameter controls"""
        row = 0
        
        # Via Diameter
        ttk.Label(parent, text="Via Diameter (mm):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.via_diameter_var = tk.DoubleVar(value=0.6)
        ttk.Spinbox(parent, from_=0.1, to=5.0, increment=0.1, format="%.2f",
                   textvariable=self.via_diameter_var, width=15,
                   validate='key', validatecommand=(self.validate_nonneg_cmd, '%P')).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # Via Drill
        ttk.Label(parent, text="Via Drill (mm):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.via_drill_entry = EntryWithPlaceholder(parent, placeholder="automatic", width=15)
        self.via_drill_entry.grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        # Via Offset
        ttk.Label(parent, text="Via Offset (mm):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.via_offset_entry = EntryWithPlaceholder(parent, placeholder="automatic", width=15)
        self.via_offset_entry.grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1
        
        # Stagger Inner Vias
        self.stagger_inner_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Stagger inner via ring", 
                       variable=self.stagger_inner_var).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1
        
        # Stagger Outer Vias
        self.stagger_outer_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Stagger outer via ring", 
                       variable=self.stagger_outer_var).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=5)
        row += 1
        
    def create_output_params(self, parent):
        row = 0

        # Footprint Name
        ttk.Label(parent, text="Footprint Name:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.footprint_name_entry = EntryWithPlaceholder(parent, placeholder="automatic")
        self.footprint_name_entry.grid(row=row, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        row += 1

        # KiCad layer names
        copper_layers = ['F.Cu', 'B.Cu'] + [f'In{i}.Cu' for i in range(1, 31)]

        # Top Layer
        ttk.Label(parent, text="Top Layer:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.top_layer_var = tk.StringVar(value="F.Cu")
        top_layer_combo = ttk.Combobox(parent, textvariable=self.top_layer_var,
                                        values=copper_layers, state='readonly', width=23)
        top_layer_combo.grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        # Bottom Layer
        ttk.Label(parent, text="Bottom Layer:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.bottom_layer_var = tk.StringVar(value="B.Cu")
        bottom_layer_combo = ttk.Combobox(parent, textvariable=self.bottom_layer_var,
                                           values=copper_layers, state='readonly', width=23)
        bottom_layer_combo.grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        # Circle Segments
        ttk.Label(parent, text="Circle Segments:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.circle_segments_entry = EntryWithPlaceholder(parent, placeholder=str(DEFAULT_CIRCLE_SEGMENTS), width=15)
        self.circle_segments_entry.grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Label(parent, text="Points per 360° for circles",
                 foreground='gray').grid(row=row, column=2, sticky=tk.W, padx=(10, 0))
        row += 1

        # Keepout Zone
        ttk.Label(parent, text="Keepout Zone:").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.keepout_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Add keepout area",
                       variable=self.keepout_var).grid(row=row, column=1, sticky=tk.W, pady=5)
        row += 1

        # Keepout Margin
        ttk.Label(parent, text="Keepout Margin (mm):").grid(row=row, column=0, sticky=tk.W, pady=5)
        self.keepout_margin_var = tk.DoubleVar(value=5.0)
        ttk.Spinbox(parent, from_=0, to=50, increment=0.5,
                   textvariable=self.keepout_margin_var, width=15,
                   validate='key', validatecommand=(self.validate_nonneg_cmd, '%P')).grid(row=row, column=1, sticky=tk.W, pady=5)
        ttk.Label(parent, text="Margin around coil",
                 foreground='gray').grid(row=row, column=2, sticky=tk.W, padx=(10, 0))
        row += 1
        
    def get_parameters(self):
        # Create the appropriate shape based on selection. It is constructed in the layout process: its skeleton can
        # take long to compute.
        shape_type = self.shape_type_var.get()

        if shape_type == "Circle":
            shape = partial(CircleShape,
                outer_diameter=self.circle_outer_dia_var.get(),
                inner_diameter=self.circle_inner_dia_var.get()
            )
        elif shape_type == "Rectangle":
            shape = partial(RectangleShape,
                width=self.rect_width_var.get(),
                height=self.rect_height_var.get(),
                annular_width=self.rect_annular_width_var.get()
            )
        elif shape_type == "Trapezoid":
            shape = partial(TrapezoidShape,
                width=self.trap_width_var.get(),
                height=self.trap_height_var.get(),
                offset=self.trap_offset_var.get(),
                annular_width=self.trap_annular_width_var.get()
            )
        elif shape_type == "Sector":
            import math
            shape = partial(SectorShape,
                inner_diameter=self.sector_inner_dia_var.get(),
                outer_diameter=self.sector_outer_dia_var.get(),
                angle=math.radians(self.sector_angle_var.get()),
                annular_width=self.sector_annular_width_var.get()
            )
        elif shape_type == "Star":
            shape = partial(StarShape,
                inner_diameter=self.star_inner_dia_var.get(),
                outer_diameter=self.star_outer_dia_var.get(),
                points=self.star_points_var.get(),
                annular_width=self.star_annular_width_var.get()
            )
        elif shape_type == "Regular Polygon":
            shape = partial(RegularPolygonShape,
                diameter=self.poly_diameter_var.get(),
                corners=self.poly_corners_var.get(),
                annular_width=self.poly_annular_width_var.get()
            )
        elif shape_type == "SVG":
            shape = partial(SVGShape,
                filename=self.svg_filename_var.get(),
                annular_width=self.svg_annular_width_var.get()
            )
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")

        params = {
            'shape'              : shape,
            'turns'              : self.turns_var.get(),
            'layers'             : self.layer_mode_var.get(),
            'twists'             : self.twists_var.get(),
            'clockwise'          : (self.direction_var.get() == "clockwise"),
            'copper_thickness'   : self.copper_thickness_var.get() / 1000.0,  # µm -> mm
            'keepout_zone'       : self.keepout_var.get(),
            'keepout_margin'     : self.keepout_margin_var.get(),
            'via_diameter'       : self.via_diameter_var.get(),
            'stagger_inner_vias' : self.stagger_inner_var.get(),
            'stagger_outer_vias' : self.stagger_outer_var.get(),
            'layer_pair'         : f"{self.top_layer_var.get()},{self.bottom_layer_var.get()}",
        }

        if (trace_width_value := self.trace_width_entry.get()):
            params['trace_width'] = float(trace_width_value)

        if (clearance_value := self.clearance_entry.get()):
            params['clearance'] = float(clearance_value)

        if (via_drill_value := self.via_drill_entry.get()):
            params['via_drill'] = float(via_drill_value)

        if (via_offset_value := self.via_offset_entry.get()):
            params['via_offset'] = float(via_offset_value)

        return params

    def get_circle_segments(self):
        circle_segments = self.circle_segments_entry.get()
        return int(circle_segments) if circle_segments else DEFAULT_CIRCLE_SEGMENTS
    
    def setup_logging(self):
        """Set up logging handler to capture kicoil logger output for display in the output text widget"""
        self.kicoil_logger = logging.getLogger('kicoil')
        self.kicoil_logger.setLevel(logging.INFO)
        self.kicoil_logger.handlers.clear()
        self.log_handler = TextWidgetHandler(self.output_text)
        self.kicoil_logger.addHandler(self.log_handler)
    
    def setup_traces(self):
        # Shape type needs special handling for tab visibility
        self.shape_type_var.trace_add('write', self.update_shape_tab_visibility)
        self.shape_type_var.trace_add('write', self._on_parameter_change)

        for var in [
                    self.turns_var,
                    self.layer_mode_var,
                    self.direction_var,
                    self.twists_var,
                    self.copper_thickness_var,
                    self.keepout_var,
                    self.keepout_margin_var,
                    self.via_diameter_var,
                    self.stagger_inner_var,
                    self.stagger_outer_var,
                    self.top_layer_var,
                    self.bottom_layer_var,
                    # Circle shape params
                    self.circle_outer_dia_var,
                    self.circle_inner_dia_var,
                    # Rectangle shape params
                    self.rect_width_var,
                    self.rect_height_var,
                    self.rect_annular_width_var,
                    # Trapezoid shape params
                    self.trap_width_var,
                    self.trap_height_var,
                    self.trap_offset_var,
                    self.trap_annular_width_var,
                    # Sector shape params
                    self.sector_outer_dia_var,
                    self.sector_inner_dia_var,
                    self.sector_angle_var,
                    self.sector_annular_width_var,
                    # Star shape params
                    self.star_outer_dia_var,
                    self.star_inner_dia_var,
                    self.star_points_var,
                    self.star_annular_width_var,
                    # Polygon shape params
                    self.poly_diameter_var,
                    self.poly_corners_var,
                    self.poly_annular_width_var,
                    # SVG shape params
                    self.svg_annular_width_var]:
            var.trace_add('write', self._on_parameter_change)

        for entry in [self.trace_width_entry, self.clearance_entry,
                      self.via_drill_entry, self.via_offset_entry,
                      self.footprint_name_entry, self.circle_segments_entry]:
            entry.bind('<KeyRelease>', lambda e: self._on_parameter_change())

        # SVG filename entry needs special handling
        self.svg_filename_var.trace_add('write', self._on_parameter_change)
    
    def _on_parameter_change(self, *args):
        # Schedule validation to avoid too many rapid calls
        if self._validation_after_id is not None:
            self.root.after_cancel(self._validation_after_id)
        self._validation_after_id = self.root.after(200, self.start_layout)
        self.current_model = None
    
    def start_layout(self):
        """ Build the model for the current parameters in a new process, stopping the one that is running """
        self._validation_after_id = None
        self.stop_layout()
        try:
            params = self.get_parameters()
        except (ValueError, tk.TclError) as e: # e.g. an empty field
            self.show_result(None, f'Invalid parameter: {e}', [])
            return
        ctx = multiprocessing.get_context('spawn') # fork is not safe with Tk in this process
        self.layout_conn, child_conn = ctx.Pipe(duplex=False)
        self.layout_process = ctx.Process(target=build_model, args=(child_conn, params), daemon=True)
        self.layout_process.start()
        child_conn.close() # so that we get an EOF if the process dies
        self.cancel_button['state'] = 'normal'
        self.status_var.set("Computing layout...")
        self.root.after(100, self.poll_layout, self.layout_process)

    def poll_layout(self, process):
        if process is not self.layout_process: # stopped or replaced
            return
        if not self.layout_conn.poll():
            self.root.after(100, self.poll_layout, process)
            return
        try:
            result = self.layout_conn.recv()
        except EOFError:
            result = None, 'The layout process ended unexpectedly (out of memory?)', []
        self.stop_layout()
        self.show_result(*result)

    def stop_layout(self):
        if self.layout_process is not None:
            self.layout_process.terminate()
            self.layout_process.join()
            self.layout_conn.close()
            self.layout_process = None
        self.cancel_button['state'] = 'disabled'
        self.status_var.set("")

    def cancel_layout(self):
        self.stop_layout()
        self.show_result(None, 'Cancelled', [])

    def show_result(self, model, error, messages):
        self.current_model = model
        self.output_text['state'] = 'normal'
        self.output_text.delete('1.0', tk.END)
        for tag, text in messages:
            self.output_text.insert(tk.END, f'{text}\n', tag)
        if error is None:
            self.output_text.insert(tk.END, "Parameters valid\n", 'info')
        else:
            self.output_text.insert(tk.END, f"ERROR: {error}\n", 'error')
        self.output_text.see(tk.END)
        self.output_text['state'] = 'disabled'
        self.update_placeholders()
        self.update_preview()

    def update_placeholders(self):
        if self.current_model is None:
            self.trace_width_entry.update_placeholder("automatic")
            self.clearance_entry.update_placeholder("automatic")
            self.via_drill_entry.update_placeholder("automatic")
            self.via_offset_entry.update_placeholder("automatic")
            self.footprint_name_entry.update_placeholder("automatic")
        
        else:
            if not self.trace_width_entry.get():
                self.trace_width_entry.update_placeholder(f"auto: {self.current_model.trace_width:.2f}")

            if not self.clearance_entry.get():
                self.clearance_entry.update_placeholder(f"auto: {self.current_model.clearance:.2f}")

            if not self.via_drill_entry.get():
                self.via_drill_entry.update_placeholder(f"auto: {self.current_model.via_drill:.2f}")

            if not self.via_offset_entry.get():
                self.via_offset_entry.update_placeholder(f"auto: {self.current_model.via_offset:.2f}")

            if not self.footprint_name_entry.get():
                self.footprint_name_entry.update_placeholder(self.current_model.default_footprint_name)

    def update_preview(self):
        if not HAS_PREVIEW:
            return

        if self.current_model is None:
            # Clear preview when model is invalid
            self.preview_canvas.delete("all")
            self.preview_raw_image = None
            self.preview_image = None
            return

        footprint = self.current_model.render_footprint(None, self.get_circle_segments())
        svg_tag = make_transparent_svg(footprint)
        viewbox = svg_tag.attrs.get('viewBox', '0 0 800 800')
        _, _, svg_width, svg_height = map(float, viewbox.split())
        min_dimension = 800
        scale = max(min_dimension / svg_width, min_dimension / svg_height)
        output_width = int(svg_width * scale)
        output_height = int(svg_height * scale)
        svg_tag.attrs['width'] = f'{output_width}px'
        svg_tag.attrs['height'] = f'{output_height}px'

        png_data = cairosvg.svg2png(bytestring=str(svg_tag).encode('utf-8'))
        self.preview_raw_image = Image.open(BytesIO(png_data))
        self._rescale_preview()

    
    def show_valid_twists(self):
        turns = self.turns_var.get()
        valid_twists = list(divisors(turns, turns))

        self.output_text['state'] = 'normal'
        self.output_text.delete('1.0', tk.END)
        self.output_text.insert('1.0', f'Valid twist counts for {turns} turns:\n')
        for d in valid_twists:
            self.output_text.insert(tk.END, f'  {d}\n')
        self.output_text['state'] = 'disabled'
    
    def update_board_footprint(self):
        """ Replace the footprint selected in KiCad's PCB editor with the coil, or place a new one if none is
        selected. """
        if self.current_model is None:
            messagebox.showerror("Error", "No layout. Please wait for it or check the output for warnings or errors.")
            return

        board = self.kicad_inst.get_board()
        selected = [item for item in board.get_selection() if isinstance(item, FootprintInstance)]
        if len(selected) > 1:
            messagebox.showerror("Error", "More than one footprint selected. Select only the footprint you want to replace.")
            return

        footprint_name = self.footprint_name_entry.get() or None
        footprint = self.current_model.render_footprint(footprint_name, self.get_circle_segments())

        if selected:
            instance, = selected
            update_footprint_instance(instance, footprint)
            board.update_items(instance)
            self.kicoil_logger.info('Updated the selected footprint on the board.')
        else:
            instance = FootprintInstance()
            instance.layer = BoardLayer.BL_F_Cu
            instance.reference_field.text.value = NEW_FOOTPRINT_REFERENCE
            update_footprint_instance(instance, footprint)
            instance, = board.create_items(instance)
            board.clear_selection()
            board.add_to_selection(instance)
            # Board.interactive_move leaves an API commit open (KiCad 10.99), so start KiCad's own move tool.
            self.kicad_inst.run_action('pcbnew.InteractiveMove.move')
            self.kicoil_logger.info('Created a new footprint. Click in the PCB editor to place it.')

    def save_footprint_file(self):
        if self.current_model is None:
            messagebox.showerror("Error", "No layout. Please wait for it or check the output for warnings or errors.")
            return
        
        try:
            footprint_name = self.footprint_name_entry.get() or None
    
            self.output_text['state'] = 'normal'
            self.output_text.insert(tk.END, "Rendering footprint...\n\n", 'info')

            footprint = self.current_model.render_footprint(footprint_name, self.get_circle_segments())
            default_name = footprint_name or self.current_model.default_footprint_name
            output_file = filedialog.asksaveasfilename(
                title="Save KiCad Footprint",
                defaultextension=".kicad_mod",
                initialfile=f"{default_name}.kicad_mod",
                filetypes=[("KiCad Footprint", "*.kicad_mod"), ("All files", "*.*")]
            )

            if not output_file:
                self.output_text.insert(tk.END, "\nSave cancelled.\n", 'info')
                return

            Path(output_file).write_text(footprint.serialize())
            self.output_text.insert(tk.END, f"\nSuccess! Footprint saved to:\n  {output_file}\n", 'info')

        except Exception as e:
            tb = traceback.format_exc()
            self.output_text['state'] = 'normal'
            self.output_text.insert(tk.END, f"\nError generating footprint:\n{tb}\n", 'error')
            self.output_text.see(tk.END)

            print(tb, file=sys.stderr)

            messagebox.showerror("Error", f"Error generating footprint: {e}")

        finally:
            self.output_text['state'] = 'disabled'


def main(kicad_inst=None):
    root = tk.Tk()
    app = KiCoilGUI(root, kicad_inst)
    root.mainloop()
