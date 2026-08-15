from fpdf import FPDF


class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, 'SEMICON India Hackathon 2026 - KLA Track', align='C', new_x='LMARGIN', new_y='NEXT')
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(0, 82, 136)
        self.cell(0, 10, title, new_x='LMARGIN', new_y='NEXT')
        self.set_draw_color(0, 82, 136)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(4)

    def sub_title(self, title):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(50, 50, 50)
        self.cell(0, 8, title, new_x='LMARGIN', new_y='NEXT')
        self.ln(2)

    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def bullet(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(40, 40, 40)
        self.cell(0, 6, '  - ' + text, new_x='LMARGIN', new_y='NEXT')

    def table_row(self, col1, col2, bold=False):
        style = 'B' if bold else ''
        self.set_font('Helvetica', style, 10)
        self.set_text_color(40, 40, 40)
        self.cell(70, 7, col1, border=1)
        self.cell(120, 7, col2, border=1, new_x='LMARGIN', new_y='NEXT')


pdf = PDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)

# ===== PAGE 1: TITLE =====
pdf.add_page()
pdf.ln(30)
pdf.set_font('Helvetica', 'B', 28)
pdf.set_text_color(0, 82, 136)
pdf.cell(0, 15, 'SEMICON India Hackathon 2026', align='C', new_x='LMARGIN', new_y='NEXT')

pdf.set_font('Helvetica', '', 16)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 10, 'KLA Track', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.ln(5)

pdf.set_font('Helvetica', 'B', 18)
pdf.set_text_color(40, 40, 40)
pdf.cell(0, 12, 'AI-Based Restoration of Degraded', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.cell(0, 12, 'Images for Semiconductor Inspection', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.ln(15)

pdf.set_font('Helvetica', '', 14)
pdf.set_text_color(100, 100, 100)
pdf.cell(0, 10, 'Team: Phantom Protocol', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.ln(5)

pdf.set_font('Helvetica', '', 12)
members = [
    'Tanmay Verma (Team Leader)',
    'Karan Agrawal',
    'Samridh Nautiyal',
    'Yuvraj Sharma',
]
for m in members:
    pdf.cell(0, 8, m, align='C', new_x='LMARGIN', new_y='NEXT')

pdf.ln(8)
pdf.set_font('Helvetica', 'I', 12)
pdf.cell(0, 8, 'SRM University, Sonipat, Haryana', align='C', new_x='LMARGIN', new_y='NEXT')

pdf.ln(20)
pdf.set_font('Helvetica', 'B', 11)
pdf.set_text_color(0, 82, 136)
pdf.cell(0, 8, 'GitHub: https://github.com/lords-coder/semicon-restoration', align='C', new_x='LMARGIN', new_y='NEXT')
pdf.set_font('Helvetica', 'I', 10)
pdf.set_text_color(150, 50, 50)
pdf.cell(0, 8, 'Demo Video: [INSERT YOUR YOUTUBE/LINK HERE]', align='C', new_x='LMARGIN', new_y='NEXT')

# ===== PAGE 2: PROBLEM & SOLUTION =====
pdf.add_page()
pdf.section_title('1. Problem Statement')
pdf.body_text(
    'Semiconductor wafers undergo visual inspection at multiple stages of manufacturing. '
    'These inspection images often suffer from noise introduced by imaging sensors, '
    'low resolution due to hardware limitations, and degradation from data compression '
    'and transmission. Manual inspection of degraded images is time-consuming, error-prone, '
    'and does not scale with modern semiconductor production volumes.'
)
pdf.body_text(
    'Objective: Build an AI model that takes degraded 128x128 noisy grayscale images '
    'and restores them to clean 256x256 high-resolution outputs, improving both visual '
    'quality and defect detection accuracy.'
)

pdf.section_title('2. Our Solution: RestoreNet')
pdf.body_text(
    'We designed RestoreNet, a lightweight convolutional neural network for single-image '
    'super-resolution and denoising, specifically tailored for semiconductor inspection images.'
)

pdf.sub_title('Architecture')
pdf.set_font('Courier', '', 9)
pdf.set_text_color(40, 40, 40)
arch_lines = [
    'Input (1x128x128)',
    '  |',
    'Conv2d(1 -> 64, 3x3)',
    '  |',
    '8x ResidualBlock (Conv -> LeakyReLU -> Conv + Skip)',
    '  |',
    'Channel Attention (AvgPool -> FC -> Sigmoid -> Scale)',
    '  |',
    'Conv2d(64 -> 64, 3x3) + Skip from first conv',
    '  |',
    'PixelShuffle 2x: Conv2d(64 -> 256) -> Reshape -> LeakyReLU',
    '  |',
    'Conv2d(64 -> 1, 3x3)',
    '  |',
    'Output (1x256x256)',
]
for line in arch_lines:
    pdf.cell(0, 5, line, new_x='LMARGIN', new_y='NEXT')
pdf.ln(5)

# ===== PAGE 3: KEY COMPONENTS =====
pdf.add_page()
pdf.section_title('3. Key Components')

pdf.table_row('Component', 'Purpose', bold=True)
pdf.table_row('Residual Blocks (x8)', 'Deep feature extraction with skip connections')
pdf.table_row('Channel Attention', 'Dynamic feature channel weighting (reduction=16)')
pdf.table_row('PixelShuffle (2x)', 'Learnable sub-pixel upsampling 128 -> 256')
pdf.table_row('Skip Connection', 'Preserves low-level features from first conv')
pdf.table_row('LeakyReLU (0.2)', 'Non-linear activation, avoids dead neurons')
pdf.ln(5)

pdf.section_title('4. Training Details')
pdf.table_row('Parameter', 'Value', bold=True)
pdf.table_row('Framework', 'PyTorch 2.12+ (CUDA 12.8)')
pdf.table_row('GPU', 'NVIDIA GeForce RTX 5060 (Laptop)')
pdf.table_row('Epochs', '450')
pdf.table_row('Batch Size', '16')
pdf.table_row('Loss Function', 'L1 Loss (Mean Absolute Error)')
pdf.table_row('Optimizer', 'Adam')
pdf.table_row('Learning Rate', '2e-4')
pdf.table_row('Data Augmentation', 'Random H-flip, V-flip, 90-deg rotation')
pdf.table_row('Checkpointing', 'Every 50 epochs')
pdf.ln(5)

pdf.section_title('5. Results')
pdf.table_row('Metric', 'Value', bold=True)
pdf.table_row('Training Image Pairs', '3,200')
pdf.table_row('Test Images', '400')
pdf.table_row('Training Epochs', '450')
pdf.table_row('PSNR (Training)', '~74+ dB')
pdf.table_row('Inference Speed', '<50ms per image (GPU)')
pdf.table_row('Model Size', '~1.2M parameters')

# ===== PAGE 4: USAGE & LINKS =====
pdf.add_page()
pdf.section_title('6. How to Use')

pdf.sub_title('Installation')
pdf.set_font('Courier', '', 9)
pdf.cell(0, 5, 'git clone https://github.com/lords-coder/semicon-restoration.git', new_x='LMARGIN', new_y='NEXT')
pdf.cell(0, 5, 'cd semicon-restoration', new_x='LMARGIN', new_y='NEXT')
pdf.cell(0, 5, 'pip install -r requirements.txt', new_x='LMARGIN', new_y='NEXT')
pdf.ln(5)

pdf.sub_title('Restore Images')
pdf.set_font('Courier', '', 9)
pdf.cell(0, 5, 'python restore_menu.py', new_x='LMARGIN', new_y='NEXT')
pdf.ln(2)
pdf.set_font('Helvetica', '', 10)
pdf.bullet('Option 1: Single image restoration (.npy, .png, .jpg)')
pdf.bullet('Option 2: Batch folder restoration')
pdf.bullet('Outputs: restored .png + side-by-side comparison')
pdf.ln(5)

pdf.sub_title('Batch Evaluation')
pdf.set_font('Courier', '', 9)
pdf.cell(0, 5, 'python evaluation.py --input_dir test_noisy_lr/NoisyLR --output_dir output/', new_x='LMARGIN', new_y='NEXT')
pdf.ln(5)

pdf.section_title('7. Dataset')
pdf.body_text(
    'Download from: Google Drive - KLA Dataset\n'
    'https://drive.google.com/drive/folders/1VKiFW-kDk9-q5XRPu3nrl08OM94EwzV6'
)
pdf.table_row('Dataset', 'Format / Size / Count', bold=True)
pdf.table_row('Ground Truth (GT)', '.npy | 256x256 uint8 | 3,200 images')
pdf.table_row('NoisyLR (Train)', '.npy | 128x128 uint8 | 3,200 images')
pdf.table_row('Test NoisyLR', '.npy | 128x128 uint8 | 400 images')
pdf.ln(5)

pdf.section_title('8. Links')
pdf.set_font('Helvetica', 'B', 10)
pdf.set_text_color(0, 82, 136)
pdf.cell(0, 8, 'GitHub Repository: https://github.com/lords-coder/semicon-restoration', new_x='LMARGIN', new_y='NEXT')
pdf.set_text_color(150, 50, 50)
pdf.cell(0, 8, 'Demo Video: [INSERT YOUR YOUTUBE/LINK HERE]', new_x='LMARGIN', new_y='NEXT')
pdf.set_text_color(0, 82, 136)
pdf.cell(0, 8, 'Hackathon Portal: https://hackathon2026.i4c.in/', new_x='LMARGIN', new_y='NEXT')
pdf.cell(0, 8, 'KLA Webinar: https://youtu.be/RMSDaviTOIw', new_x='LMARGIN', new_y='NEXT')

# Save
output_path = 'C:/hackathon_project/SEMICON_KLA_Submission_PhantomProtocol.pdf'
pdf.output(output_path)
print(f'PDF saved to: {output_path}')
