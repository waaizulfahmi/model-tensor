## README for Python Project

### Face Recoginition

[Face Recoginition]

### Project Description

[klasifikasi muka yang di kirim menggunakan http, untuk keperluan ]

### Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

### Installation

Add variable environment

- OPENBLAS_NUM_THREADS = 1
- LOKY_MAX_CPU_COUNT = 4
- OMP_NUM_THREADS = 1

```bash
# Clone the repository
git clone [repository-url]

# Navigate to the project directory
cd [project-directory]

# Install dependencies (if any)
pip install -r requirements.txt

```

setelah itu uncomment app.py bagian ini
[if __name__ == "__main__":
app.run(debug=True)]

```bash
# Run the project
python app.py
```

### Usage

[menggunakan endpoint]

- /login POST
- /register POST
- /test GET

### Features

- register: [untuk register wajah yang akan di daftarkan]
- login: [untuk mengklasifikasi wajah]
- test: [untuk menampilkan isi data model]

### Contributing

[Jelaskan bagaimana orang lain dapat berkontribusi pada proyek Anda. Sertakan pedoman untuk mengirimkan masalah, permintaan penarikan, atau standar pengkodean apa pun yang Anda ingin agar diikuti oleh kontributor.]

1. Fork proyek.
2. Buat cabang baru (`git checkout -b feature/nama-fitur`).
3. Buat perubahan dan komit (`git commit -m 'Tambahkan fitur baru'`).
4. Dorong ke cabang (`git push origin feature/feature-name`).
5. Buka permintaan tarik.
