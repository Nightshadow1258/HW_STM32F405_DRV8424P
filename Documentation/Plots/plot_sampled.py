import pandas as pd
import matplotlib.pyplot as plt
import yaml
import os
import sys
import re

def print_usage():
    print("""
Usage:
    python plot_single.py <config.yaml>

Description:
    This script reads one or more oscilloscope CSV files and plots selected channels based on settings in a YAML config file.

Input File Options (choose one):
    input_file:           Path to a single CSV file (e.g., input/data.csv)
    input_files:          List of multiple CSV files to process (e.g., ["input/data1.csv", "input/data2.csv"])
                          If neither is specified, all CSV files in the 'input/' folder will be processed automatically.

Available Configuration Options in config.yaml:
    channels_to_plot:     List of channels to plot (e.g., ["CH1"], ["CH2"], or ["CH1", "CH2"]).
                          Leave empty [] to plot all available channels.
    extract_legend_from_filename:
                          If true, extracts parenthesized text from filename for legend labels.
    plot_format:
        size:             Plot dimensions in inches [width, height] (e.g., {width: 12, height: 7.5})
        font_sizes:       Font size settings for title, legend, axis, and ticks.
        grid:             Custom grid colors and linestyles.
        axis_labels:      Labels for x and y axes.
        plot_title:       Title string displayed on the plot.
    time_window:          Time range to display [start_s, end_s] in seconds.
                          Leave empty or omit to plot full range.
    header_cleanup:       List of substrings to remove from column headers (e.g., ["Ave. (C)", "(C)"])
    save_formats:         Output formats to save plot image or data (e.g., image: ["jpeg", "pdf"])
    output_dir:           Optional. Directory to save output files. Defaults to "output/"

Example:
    python plot_single.py config.yaml
""")

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def clean_column_names(columns, cleanup_rules):
    cleaned = []
    for col in columns:
        for rule in cleanup_rules:
            col = col.replace(rule, "")
        cleaned.append(col.strip())
    return cleaned

def get_input_files(config):
    # Priority: input_files > input_file > all CSVs in input/
    if "input_files" in config:
        files = config["input_files"]
        return files if isinstance(files, list) else [files]
    elif "input_file" in config:
        files = config["input_file"]
        return files if isinstance(files, list) else [files]
    else:
        return [os.path.join("input", f) for f in os.listdir("input") if f.endswith(".csv")]

def parse_oscilloscope_csv(csv_file):
    """
    Parses Siglent oscilloscope CSV format.
    Skips top metadata header block (11 lines) and reads:
      - Column index 3: Time [s] ('Second')
      - Column index 4: CH1 Voltage [V] ('Volt')
      - Column index 5: CH2 Voltage [V] ('Volt')
    """
    df_raw = pd.read_csv(csv_file, skiprows=11, header=None)

    # Extract relevant data columns into a structured DataFrame
    df_clean = pd.DataFrame({
        'Time': pd.to_numeric(df_raw.iloc[:, 3], errors='coerce'),
        'CH1': pd.to_numeric(df_raw.iloc[:, 4], errors='coerce'),
        'CH2': pd.to_numeric(df_raw.iloc[:, 5], errors='coerce')
    }).dropna()

    return df_clean

def get_legend_mappings_from_filename(filename):
    """
    Extracts labels inside parentheses from filename and maps them to CH1, CH2, etc.
    Example: 'DRV8424_9V_(VM)_(A1)' -> {'CH1': 'VM', 'CH2': 'A1'}
    """
    matches = re.findall(r'\((.*?)\)', filename)
    mappings = {}
    for i, label in enumerate(matches, start=1):
        mappings[f"CH{i}"] = label
    return mappings

def auto_scale_time(time_series):
    """
    Automatically scales time vector to optimal unit (ns, us, ms, s) based on total time span.
    Returns: scaled_time_series, unit_label_string
    """
    if time_series.empty:
        return time_series, "Time [s]"

    time_span = time_series.max() - time_series.min()

    if time_span < 1e-6:
        return time_series * 1e9, "Time [ns]"
    elif time_span < 1e-3:
        return time_series * 1e6, "Time [µs]"
    elif time_span < 1.0:
        return time_series * 1e3, "Time [ms]"
    else:
        return time_series, "Time [s]"

def apply_plot_formatting(config, time_unit_label="Time [s]"):
    plot_format = config.get("plot_format", {})
    fonts = plot_format.get("font_sizes", {})

    # Priority: Auto scaled unit label > manual config label > fallback default
    x_label = time_unit_label if time_unit_label else plot_format.get("axis_labels", {}).get("x", "Time [s]")

    plt.xlabel(x_label, fontsize=fonts.get("axis", 14))
    plt.ylabel(plot_format.get("axis_labels", {}).get("y", "Y-AXIS NAME PLACEHOLDER"), fontsize=fonts.get("axis", 14))
    plt.title(plot_format.get("plot_title", "TITLE"), fontsize=fonts.get("title", 16))
    plt.legend(fontsize=fonts.get("legend", 12))

    major = plot_format.get("grid", {}).get("major", {})
    minor = plot_format.get("grid", {}).get("minor", {})
    
    # Fallback default colors if alpha set to 00
    major_color = major.get("color", "#808080")
    minor_color = minor.get("color", "#A8A8A8")
    if major_color.endswith("00"):
        major_color = "#808080"
    if minor_color.endswith("00"):
        minor_color = "#A8A8A8"

    plt.grid(True, which='major', axis='both', 
             color=major_color,
             linestyle=major.get("linestyle", ":"),
             linewidth=major.get("linewidth", 0.8))
    plt.grid(True, which='minor', axis='both', 
             color=minor_color,
             linestyle=minor.get("linestyle", ":"),
             linewidth=minor.get("linewidth", 0.5))

    plt.minorticks_on()
    plt.xticks(fontsize=fonts.get("ticks", 12))
    plt.yticks(fontsize=fonts.get("ticks", 12))

    plt.tight_layout()

def save_formats(plt, df, csv_file, config, output_tag=""):
    """
    Saves plot and cleaned data in multiple formats as specified in the config.

    Parameters:
    - plt (matplotlib.pyplot): The plotting context containing the figure to be saved.
    - df (pandas.DataFrame): The cleaned DataFrame to be exported.
    - csv_file (str): Path to the original CSV file, used to derive the output filename.
    - config (dict): Configuration dictionary containing:
        - 'output_dir' (str): Directory where outputs will be saved. Defaults to 'output'.
        - 'save_formats' (dict): Specifies formats to save:
            - 'image' (list[str]): List of image formats (e.g., ['jpeg', 'pdf']) for plot export.
            - 'data' (list[str]): List of data formats (e.g., ['csv', 'xlsx']) for DataFrame export.

    Behavior:
    - Creates the output directory if it doesn't exist.
    - Saves the plot in each specified image format using `plt.savefig`.
    - Saves the DataFrame in each specified data format using `df.to_csv` or `df.to_excel`.

    Example config:
    save_formats:
      image: ["jpeg", "pdf"]
      data: ["csv", "xlsx"]
    """

    filename = os.path.splitext(os.path.basename(csv_file))[0]
    output_dir = config.get("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)
    save_format_config = config.get("save_formats", {})

    for fmt_img in save_format_config.get("image", []) or []:
        plt.savefig(f"{output_dir}/{filename}{output_tag}.{fmt_img}", format=fmt_img, bbox_inches='tight')

    for fmt_data in save_format_config.get("data", []) or []:
        if fmt_data == "csv":
            df.to_csv(f"{output_dir}/{filename}{output_tag}_cleaned.csv", index=False)
        elif fmt_data == "xlsx":
            df.to_excel(f"{output_dir}/{filename}{output_tag}_cleaned.xlsx", index=False)

def main():
    if len(sys.argv) != 2 or sys.argv[1] in ("--help", "-h"):
        print_usage()
        return

    config_file = sys.argv[1]
    config = load_config(config_file)
    input_files = get_input_files(config)

    # determine if batch mode
    batch_mode = len(input_files) > 1

    for csv_file in input_files:
        print(f"Processing: {csv_file}")
        
        # Load and parse oscilloscope CSV file directly using recorded time vectors
        df = parse_oscilloscope_csv(csv_file)
        cleanup_rules = config.get("header_cleanup", [])
        df.columns = clean_column_names(df.columns, cleanup_rules)

        # Extract legend labels from filename if enabled
        filename = os.path.splitext(os.path.basename(csv_file))[0]
        legend_mappings = {}
        if config.get("extract_legend_from_filename", True):
            legend_mappings = get_legend_mappings_from_filename(filename)

        # Time window filter using time values in seconds
        time_window = config.get("time_window", [])
        if time_window and len(time_window) == 2:
            start_s, end_s = time_window
            df_filtered = df[(df['Time'] >= start_s) & (df['Time'] <= end_s)].copy()
        else:
            df_filtered = df.copy()

        # Automatic time scaling (convert to ns, us, ms, or s)
        time_scaled, time_label = auto_scale_time(df_filtered['Time'])
        df_filtered['Time_Scaled'] = time_scaled

        # Channel selection: use channels specified in config or plot all available channels
        channels_to_plot = config.get("channels_to_plot", [])
        if not channels_to_plot:
            channels_to_plot = [col for col in df.columns if col not in ('Time', 'Time_Scaled')]

        # Plot setup
        plot_format = config.get("plot_format", {})
        plot_format_size = plot_format.get("size", {})
        plt.figure(figsize=[plot_format_size.get("width", 12), plot_format_size.get("height", 7.5)])

        # Plot each selected channel against the dynamically scaled time axis
        for ch in channels_to_plot:
            if ch in df_filtered.columns:
                # Use extracted legend label if available, otherwise default to channel name
                plot_label = legend_mappings.get(ch, ch)
                plt.plot(df_filtered['Time_Scaled'], df_filtered[ch], label=plot_label)

        # apply formatting with auto scaled x-axis label
        apply_plot_formatting(config, time_unit_label=time_label)

        if not df_filtered.empty:
            plt.xlim(df_filtered['Time_Scaled'].iloc[0], df_filtered['Time_Scaled'].iloc[-1])

        # SAVE FIRST before displaying or closing the figure context
        save_formats(plt, df_filtered, csv_file, config)

        if not batch_mode:
            plt.show()
        
        plt.close('all')

if __name__ == "__main__":
    main()