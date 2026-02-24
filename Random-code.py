# Random-code



# Program Manajemen Nilai Siswa

import statistics

class Siswa:
    def __init__(self, nama):
        self.nama = nama
        self.nilai = []

    def tambah_nilai(self, nilai):
        self.nilai.append(nilai)

    def rata_rata(self):
        return statistics.mean(self.nilai)

# Input
nama = input("Masukkan nama siswa: ")
siswa = Siswa(nama)

# Loop input nilai
while True:
    try:
        n = input("Masukkan nilai (atau ketik 'stop'): ")

        if n.lower() == "stop":
            break

        nilai = float(n)

        if nilai < 0 or nilai > 100:
            print("Nilai harus 0–100")
            continue

        siswa.tambah_nilai(nilai)

    except ValueError:
        print("Input tidak valid")

# Output
if len(siswa.nilai) > 0:
    print("\nNama:", siswa.nama)
    print("Daftar nilai:", siswa.nilai)
    print("Rata-rata:", siswa.rata_rata())
else:
    print("Tidak ada nilai dimasukkan")



"""
PROGRAM PYTHON SUPERSET
Mewakili seluruh kategori fitur inti Python
"""

# ===== IMPORT =====
import math
import sys
import threading
import asyncio
from functools import reduce
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

# ===== ENUM =====
class Status(Enum):
    ACTIVE = 1
    INACTIVE = 0

# ===== DECORATOR =====
def logger(func):
    def wrapper(*args, **kwargs):
        print(f"[LOG] Memanggil {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

# ===== FUNCTION =====
@logger
def tambah(a: int, b: int) -> int:
    return a + b

# ===== LAMBDA =====
kali = lambda x, y: x * y

# ===== GENERATOR =====
def hitung(n):
    for i in range(n):
        yield i * i

# ===== DATACLASS =====
@dataclass
class User:
    nama: str
    umur: int
    status: Status = Status.ACTIVE

# ===== OOP =====
class Hewan:
    def suara(self):
        raise NotImplementedError

class Kucing(Hewan):
    def suara(self):
        return "Meong"

# ===== EXCEPTION =====
def bagi(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        return None
    finally:
        pass

# ===== FILE HANDLING =====
def simpan_file(text):
    with open("output.txt", "w") as f:
        f.write(text)

# ===== FUNCTIONAL =====
angka = [1, 2, 3, 4]
kuadrat = list(map(lambda x: x**2, angka))
genap = list(filter(lambda x: x % 2 == 0, angka))
total = reduce(lambda x, y: x + y, angka)

# ===== THREADING =====
def thread_task():
    print("Thread berjalan")

thread = threading.Thread(target=thread_task)

# ===== ASYNC =====
async def async_task():
    await asyncio.sleep(1)
    return "Async selesai"

# ===== MAIN =====
def main():
    print(tambah(2, 3))
    print(kali(4, 5))
    print(list(hitung(5)))

    user = User("Ramadani", 17)
    print(user)

    k = Kucing()
    print(k.suara())

    print(bagi(10, 0))
    simpan_file("Hello Python")

    thread.start()
    thread.join()

    print(kuadrat, genap, total)

    print(sys.argv)

    asyncio.run(async_task())

# ===== ENTRY POINT =====
if __name__ == "__main__":
    main()



"""
File ini secara sadar memuat SELURUH keyword Python resmi (Python 3.12)
Sebagian keyword berada dalam konteks non-eksekusi demi validitas sintaks.
"""

# ===== import, from, as =====
import math as m
from math import sqrt as akar

# ===== global =====
x_global = 10

# ===== class =====
class Contoh:
    def __init__(self):
        self.nilai = True

    def metode(self):
        return None


# ===== def, return, pass =====
def fungsi(a, b):
    pass
    return a and b


# ===== lambda =====
fungsi_lambda = lambda x: x or False


# ===== if, elif, else, not, is, in =====
if True is True and 1 in [1, 2, 3]:
    hasil = "benar"
elif not False:
    hasil = "masih benar"
else:
    hasil = "salah"


# ===== for, break, continue =====
for i in range(3):
    if i == 1:
        continue
    if i == 2:
        break


# ===== while =====
while False:
    pass


# ===== try, except, finally, raise, assert =====
try:
    assert True
    if False:
        raise ValueError("Tidak terjadi")
except ValueError as e:
    print(e)
finally:
    selesai = True


# ===== with =====
with open(__file__, "r") as f:
    isi = f.readline()


# ===== del =====
del isi


# ===== async, await =====
async def fungsi_async():
    await fungsi_async_kecil()
    return True

async def fungsi_async_kecil():
    return False


# ===== yield =====
def generator():
    yield 1
    yield 2


# ===== nonlocal =====
def luar():
    nilai = 10
    def dalam():
        nonlocal nilai
        nilai += 1
        return nilai
    return dalam()


# ===== match, case =====
def cocokkan(x):
    match x:
        case 1:
            return "satu"
        case 2:
            return "dua"
        case _:
            return "lainnya"


# ===== from, import, as (ulang, legal) =====
from math import pi as PI


# ===== if __name__ =====
if __name__ == "__main__":
    obj = Contoh()
    print(obj.metode())
    print(fungsi(True, False))
    print(fungsi_lambda(True))
    print(list(generator()))
    print(luar())
    print(cocokkan(2))



"""
SCRIPT VERIFIKASI KEYWORD PYTHON

Fungsi:
- Mengambil keyword resmi dari interpreter Python
- Membaca file Python target
- Memverifikasi apakah SELURUH keyword ada di file tersebut
"""

import keyword
import sys
import re
from pathlib import Path


def ekstrak_kata_kunci(teks: str) -> set:
    """
    Mengambil semua token kata dari teks Python
    (tanpa mengeksekusi kodenya)
    """
    return set(re.findall(r"\b[a-zA-Z_]+\b", teks))


def verifikasi_keyword(file_path: str):
    # Keyword resmi Python (sesuai interpreter)
    keyword_resmi = set(keyword.kwlist)

    # Baca file Python target
    kode = Path(file_path).read_text(encoding="utf-8")

    # Ambil semua kata di file
    token_file = ekstrak_kata_kunci(kode)

    # Keyword yang ditemukan & yang hilang
    ditemukan = keyword_resmi & token_file
    hilang = keyword_resmi - token_file

    print("=== HASIL VERIFIKASI KEYWORD PYTHON ===")
    print(f"Total keyword resmi : {len(keyword_resmi)}")
    print(f"Ditemukan di file  : {len(ditemukan)}")
    print(f"Hilang             : {len(hilang)}\n")

    if not hilang:
        print("✅ SEMUA KEYWORD PYTHON ADA DI FILE")
    else:
        print("❌ KEYWORD YANG HILANG:")
        for kw in sorted(hilang):
            print("-", kw)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_python_keywords.py <file.py>")
        sys.exit(1)

    verifikasi_keyword(sys.argv[1])



"""
AUTO-GENERATOR FILE KEYWORD SUPERSET PYTHON

Menghasilkan satu file Python yang memuat SELURUH keyword resmi
berdasarkan interpreter Python yang sedang digunakan.
"""

import keyword
from pathlib import Path
from datetime import datetime


OUTPUT_FILE = "python_keyword_superset.py"


def generate():
    kw = sorted(keyword.kwlist)

    now = datetime.now().isoformat()

    content = []

    # ===== HEADER =====
    content.append(f'''"""
AUTO-GENERATED PYTHON KEYWORD SUPERSET
Generated at : {now}
Python ver   : {keyword.__file__}

TOTAL KEYWORDS: {len(kw)}
{", ".join(kw)}
"""
''')

    # ===== BASIC LITERALS =====
    content.append("# === Literals ===")
    content.append("flag_true = True")
    content.append("flag_false = False")
    content.append("nilai_none = None\n")

    # ===== IMPORT =====
    content.append("# === import / from / as ===")
    content.append("import math as m")
    content.append("from math import sqrt as akar\n")

    # ===== GLOBAL =====
    content.append("# === global ===")
    content.append("nilai_global = 1\n")

    # ===== CLASS / DEF =====
    content.append("# === class / def / return / pass ===")
    content.append("""
class Contoh:
    def metode(self):
        pass
        return nilai_global
""")

    # ===== LAMBDA =====
    content.append("# === lambda ===")
    content.append("fungsi_lambda = lambda x: x\n")

    # ===== IF / ELIF / ELSE / LOGIC ===
    content.append("# === if / elif / else / and / or / not / is / in ===")
    content.append("""
if True and not False and 1 in [1, 2, 3] and None is None:
    hasil = True
elif False or True:
    hasil = False
else:
    hasil = None
""")

    # ===== FOR / WHILE =====
    content.append("# === for / while / break / continue ===")
    content.append("""
for i in range(3):
    if i == 1:
        continue
    if i == 2:
        break

while False:
    pass
""")

    # ===== TRY / EXCEPT / FINALLY / RAISE / ASSERT ===
    content.append("# === try / except / finally / raise / assert ===")
    content.append("""
try:
    assert True
    if False:
        raise ValueError("Tidak terjadi")
except ValueError as e:
    error = e
finally:
    selesai = True
""")

    # ===== WITH =====
    content.append("# === with ===")
    content.append("""
with open(__file__, "r") as f:
    baris = f.readline()
""")

    # ===== DEL =====
    content.append("# === del ===")
    content.append("del baris\n")

    # ===== ASYNC / AWAIT =====
    content.append("# === async / await ===")
    content.append("""
async def tugas_async():
    await tugas_async_kecil()
    return True

async def tugas_async_kecil():
    return False
""")

    # ===== YIELD =====
    content.append("# === yield ===")
    content.append("""
def generator():
    yield 1
    yield 2
""")

    # ===== NONLOCAL =====
    content.append("# === nonlocal ===")
    content.append("""
def luar():
    nilai = 10
    def dalam():
        nonlocal nilai
        nilai += 1
        return nilai
    return dalam()
""")

    # ===== MATCH / CASE (Python 3.10+) =====
    content.append("# === match / case ===")
    content.append("""
def cocokkan(x):
    match x:
        case 1:
            return "satu"
        case _:
            return "lain"
""")

    # ===== MAIN =====
    content.append("# === __main__ ===")
    content.append("""
if __name__ == "__main__":
    print("Keyword superset aktif")
""")

    # ===== WRITE FILE =====
    Path(OUTPUT_FILE).write_text("".join(content), encoding="utf-8")

    print(f"✅ File '{OUTPUT_FILE}' berhasil dibuat")
    print(f"📦 Total keyword: {len(kw)}")
    print("🔎 Gunakan script verifikasi untuk pengecekan")


if __name__ == "__main__":
    generate()



import ast
import keyword
from pathlib import Path
from datetime import datetime

OUTPUT = "python_ast_keyword_superset.py"

TEMPLATE = '''
def _func():
    pass
'''

def generate():
    code = []
    kw = sorted(keyword.kwlist)

    code.append(f'''"""
AST-AWARE KEYWORD SUPERSET
Generated: {datetime.now().isoformat()}
Keywords: {len(kw)}
"""
''')

    # === Literal keywords ===
    code.append("a = True\nb = False\nc = None\n")

    # === Flow & logic ===
    code.append("""
if True and not False or True:
    pass
""")

    # === Loop keywords ===
    code.append("""
for i in range(3):
    if i == 1:
        continue
    break

while False:
    pass
""")

    # === Try block ===
    code.append("""
try:
    assert True
except Exception as e:
    error = e
finally:
    pass
""")

    # === Function keywords ===
    code.append("""
def fungsi(a, b):
    global x
    x = a + b
    return x
""")

    # === Lambda ===
    code.append("f = lambda x: x\n")

    # === Class ===
    code.append("""
class A:
    def m(self):
        return True
""")

    # === With ===
    code.append("""
with open(__file__, "r") as f:
    data = f.readline()
del data
""")

    # === Generator ===
    code.append("""
def gen():
    yield 1
""")

    # === Nonlocal ===
    code.append("""
def outer():
    v = 1
    def inner():
        nonlocal v
        return v
    return inner()
""")

    # === Async ===
    code.append("""
async def coro():
    await coro2()

async def coro2():
    return None
""")

    # === Match / case ===
    code.append("""
def match_fn(x):
    match x:
        case 1:
            return True
        case _:
            return False
""")

    # === Import keywords ===
    code.append("""
import math as m
from math import sqrt as s
""")

    # === Validate AST ===
    full_code = "".join(code)
    ast.parse(full_code)  # raises if invalid

    Path(OUTPUT).write_text(full_code, encoding="utf-8")
    print(f"✅ AST-valid keyword superset generated: {OUTPUT}")

if __name__ == "__main__":
    generate()



from pathlib import Path
import keyword

OUTPUT = "python_bnf_ordered_keywords.txt"

GRAMMAR_MAP = {
    "LITERALS": {"True", "False", "None"},
    "LOGICAL": {"and", "or", "not", "is", "in"},
    "IMPORT": {"import", "from", "as"},
    "CONTROL_FLOW": {"if", "elif", "else", "for", "while", "break", "continue", "pass"},
    "DEFINITION": {"def", "class", "lambda", "return"},
    "SCOPE": {"global", "nonlocal"},
    "EXCEPTION": {"try", "except", "finally", "raise", "assert"},
    "ASYNC": {"async", "await"},
    "CONTEXT": {"with"},
    "GENERATOR": {"yield"},
    "PATTERN": {"match", "case"},
}

def generate():
    kw = set(keyword.kwlist)
    used = set()

    lines = []
    for category, words in GRAMMAR_MAP.items():
        found = sorted(words & kw)
        used |= set(found)
        lines.append(f"[{category}]")
        lines.extend(found)
        lines.append("")

    remaining = sorted(kw - used)
    if remaining:
        lines.append("[UNCLASSIFIED]")
        lines.extend(remaining)

    Path(OUTPUT).write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ BNF-ordered keyword list generated: {OUTPUT}")

if __name__ == "__main__":
    generate()



import ast
import keyword

TEMPLATES = {
    "top_level": "{kw}",
    "function": "def f():\n    {kw}",
    "loop": "for i in range(1):\n    {kw}",
    "async": "async def f():\n    {kw}",
    "class": "class A:\n    {kw}",
}

def test_keyword(kw):
    results = {}
    for ctx, tpl in TEMPLATES.items():
        try:
            code = tpl.format(kw=kw)
            ast.parse(code)
            results[ctx] = "VALID"
        except SyntaxError:
            results[ctx] = "INVALID"
    return results

def main():
    for kw in keyword.kwlist:
        print(f"\nKEYWORD: {kw}")
        res = test_keyword(kw)
        for ctx, status in res.items():
            print(f"  {ctx:10}: {status}")

if __name__ == "__main__":
    main()



import ast
from collections import defaultdict

class CFGNode:
    def __init__(self, name):
        self.name = name
        self.next = []

    def connect(self, node):
        self.next.append(node)

class CFGBuilder(ast.NodeVisitor):
    def __init__(self):
        self.counter = 0
        self.graph = defaultdict(list)
        self.current = self.new_node("start")

    def new_node(self, label):
        self.counter += 1
        return f"{label}_{self.counter}"

    def visit(self, node):
        node_name = self.new_node(type(node).__name__)
        self.graph[self.current].append(node_name)
        prev = self.current
        self.current = node_name
        super().visit(node)
        self.current = prev

    def visit_If(self, node):
        if_node = self.new_node("if")
        self.graph[self.current].append(if_node)

        # true branch
        self.current = if_node
        for stmt in node.body:
            self.visit(stmt)

        true_end = self.current

        # false branch
        self.current = if_node
        for stmt in node.orelse:
            self.visit(stmt)

        false_end = self.current

        merge = self.new_node("merge")
        self.graph[true_end].append(merge)
        self.graph[false_end].append(merge)

        self.current = merge

    def build(self, code):
        tree = ast.parse(code)
        self.visit(tree)
        return self.graph


def print_cfg(cfg):
    print("CONTROL FLOW GRAPH")
    for src, dsts in cfg.items():
        for d in dsts:
            print(f"{src} → {d}")


if __name__ == "__main__":
    code = """
def f(x):
    if x:
        return 1
    else:
        return 2
"""
    builder = CFGBuilder()
    cfg = builder.build(code)
    print_cfg(cfg)



import ast
import keyword
from pathlib import Path

OUTPUT = "minimal_code_per_keyword.txt"

TEMPLATES = {
    "True": "x = True",
    "False": "x = False",
    "None": "x = None",

    "and": "x = True and False",
    "or": "x = True or False",
    "not": "x = not False",

    "if": "if True:\n    pass",
    "elif": "if False:\n    pass\nelif True:\n    pass",
    "else": "if False:\n    pass\nelse:\n    pass",

    "for": "for i in range(1):\n    pass",
    "while": "while False:\n    pass",
    "break": "for i in range(1):\n    break",
    "continue": "for i in range(1):\n    continue",
    "pass": "pass",

    "def": "def f():\n    pass",
    "return": "def f():\n    return",
    "lambda": "f = lambda x: x",

    "class": "class A:\n    pass",

    "try": "try:\n    pass\nexcept:\n    pass",
    "except": "try:\n    pass\nexcept:\n    pass",
    "finally": "try:\n    pass\nfinally:\n    pass",
    "raise": "raise Exception()",
    "assert": "assert True",

    "with": "with open(__file__, 'r') as f:\n    pass",
    "as": "with open(__file__, 'r') as f:\n    pass",

    "import": "import math",
    "from": "from math import sqrt",

    "global": "def f():\n    global x",
    "nonlocal": "def f():\n    x=1\n    def g():\n        nonlocal x",

    "is": "x = None is None",
    "in": "x = 1 in [1]",

    "async": "async def f():\n    pass",
    "await": "async def f():\n    await g()\nasync def g():\n    pass",

    "yield": "def f():\n    yield 1",

    "match": "match 1:\n    case 1:\n        pass",
    "case": "match 1:\n    case 1:\n        pass",

    "del": "x = 1\ndel x",
}

def validate(code):
    ast.parse(code)

def main():
    lines = []
    for kw in keyword.kwlist:
        code = TEMPLATES.get(kw)
        if not code:
            lines.append(f"{kw}: ❌ NO TEMPLATE")
            continue
        try:
            validate(code)
            lines.append(f"{kw}: ✅ VALID")
            lines.append(code)
        except SyntaxError as e:
            lines.append(f"{kw}: ❌ INVALID ({e})")

        lines.append("-" * 40)

    Path(OUTPUT).write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Generated minimal code list: {OUTPUT}")

if __name__ == "__main__":
    main()



from collections import defaultdict
import ast

class BasicBlock:
    def __init__(self, name):
        self.name = name
        self.stmts = []
        self.next = []

class CFG:
    def __init__(self):
        self.blocks = {}
        self.start = self.new_block("entry")

    def new_block(self, name):
        block = BasicBlock(name)
        self.blocks[name] = block
        return block

class SSAConverter:
    def __init__(self):
        self.version = defaultdict(int)
        self.env = {}

    def new_var(self, name):
        self.version[name] += 1
        return f"{name}_{self.version[name]}"

    def convert_stmt(self, stmt):
        if isinstance(stmt, ast.Assign):
            target = stmt.targets[0].id
            new = self.new_var(target)
            self.env[target] = new
            return f"{new} = <expr>"
        elif isinstance(stmt, ast.Return):
            return f"return {self.env.get(stmt.value.id, stmt.value.id)}"
        return "<stmt>"

    def convert_block(self, block):
        ssa = []
        for stmt in block.stmts:
            ssa.append(self.convert_stmt(stmt))
        return ssa

def example_cfg():
    cfg = CFG()
    then = cfg.new_block("then")
    els = cfg.new_block("else")
    merge = cfg.new_block("merge")

    cfg.start.stmts.append(ast.Assign([ast.Name("x")], ast.Constant(0)))
    cfg.start.next = [then, els]

    then.stmts.append(ast.Assign([ast.Name("x")], ast.Constant(1)))
    then.next = [merge]

    els.stmts.append(ast.Assign([ast.Name("x")], ast.Constant(2)))
    els.next = [merge]

    merge.stmts.append(ast.Return(ast.Name("x")))

    return cfg

if __name__ == "__main__":
    cfg = example_cfg()
    ssa = SSAConverter()

    for name, block in cfg.blocks.items():
        print(f"\nBLOCK {name}")
        for line in ssa.convert_block(block):
            print(" ", line)

    print("\nφ(x_1, x_2) → x_3  (conceptual)")



def cfg_to_dot(cfg):
    lines = ["digraph CFG {"]
    for block in cfg.blocks.values():
        label = "\\l".join(block.stmts) + "\\l"
        lines.append(f'{block.name} [shape=box,label="{block.name}\\l{label}"];')
        for nxt in block.next:
            lines.append(f"{block.name} -> {nxt.name};")
    lines.append("}")
    return "\n".join(lines)



import ast
import dis

code = """
def f(x):
    if x:
        return 1
    else:
        return 2
"""

tree = ast.parse(code)
compiled = compile(tree, "<ast>", "exec")

print("=== BYTECODE ===")
dis.dis(compiled)



import dis

def bytecode_cfg(func):
    instructions = list(dis.get_instructions(func))
    edges = []

    for i, instr in enumerate(instructions):
        if instr.opname.startswith("JUMP") or "JUMP" in instr.opname:
            edges.append((instr.offset, instr.argval))
        if instr.opname == "RETURN_VALUE":
            edges.append((instr.offset, "EXIT"))
        if i + 1 < len(instructions):
            edges.append((instr.offset, instructions[i+1].offset))

    return edges

def f(x):
    if x:
        return 1
    else:
        return 2

cfg = bytecode_cfg(f)

print("BYTECODE CFG EDGES")
for e in cfg:
    print(e)



class LoopDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.visited = set()
        self.stack = set()
        self.loops = []

    def dfs(self, block):
        self.visited.add(block)
        self.stack.add(block)

        for nxt in block.next:
            if nxt not in self.visited:
                self.dfs(nxt)
            elif nxt in self.stack:
                self.loops.append((block.name, nxt.name))

        self.stack.remove(block)

    def detect(self):
        self.dfs(self.cfg.start)
        return self.loops



import dis
from collections import defaultdict

class BytecodeSSA:
    def __init__(self):
        self.version = defaultdict(int)
        self.env = {}
        self.stack = []

    def new(self, name):
        self.version[name] += 1
        return f"{name}_{self.version[name]}"

    def convert(self, func):
        ssa = []
        for instr in dis.get_instructions(func):
            op = instr.opname

            if op == "LOAD_FAST":
                v = self.env[instr.argval]
                self.stack.append(v)

            elif op == "LOAD_CONST":
                t = self.new("const")
                ssa.append(f"{t} = {instr.argval}")
                self.stack.append(t)

            elif op == "STORE_FAST":
                val = self.stack.pop()
                v = self.new(instr.argval)
                self.env[instr.argval] = v
                ssa.append(f"{v} = {val}")

            elif op == "BINARY_ADD":
                b = self.stack.pop()
                a = self.stack.pop()
                t = self.new("tmp")
                ssa.append(f"{t} = {a} + {b}")
                self.stack.append(t)

            elif op == "RETURN_VALUE":
                val = self.stack.pop()
                ssa.append(f"return {val}")

        return ssa



def dce(ssa):
    used = set()

    for line in ssa:
        for token in line.split():
            if "_" in token:
                used.add(token.strip("=+"))

    optimized = []
    for line in ssa:
        lhs = line.split("=")[0].strip()
        if lhs.startswith("return") or lhs in used:
            optimized.append(line)

    return optimized



def cse(ssa):
    seen = {}
    optimized = []

    for line in ssa:
        if "=" in line:
            lhs, rhs = map(str.strip, line.split("=", 1))
            if rhs in seen:
                optimized.append(f"{lhs} = {seen[rhs]}")
            else:
                seen[rhs] = lhs
                optimized.append(line)
        else:
            optimized.append(line)

    return optimized



def optimize(func):
    ssa = BytecodeSSA().convert(func)

    print("\n=== ORIGINAL SSA ===")
    for l in ssa: print(l)

    ssa = cse(ssa)
    print("\n=== AFTER CSE ===")
    for l in ssa: print(l)

    ssa = dce(ssa)
    print("\n=== AFTER DCE ===")
    for l in ssa: print(l)



def compute_dominators(cfg):
    blocks = list(cfg.blocks.values())
    start = cfg.start

    dom = {b: set(blocks) for b in blocks}
    dom[start] = {start}

    changed = True
    while changed:
        changed = False
        for b in blocks:
            if b is start:
                continue
            preds = [p for p in blocks if b in p.next]
            new_dom = {b}.union(set.intersection(*(dom[p] for p in preds)))
            if new_dom != dom[b]:
                dom[b] = new_dom
                changed = True

    return dom



def register_allocate(ssa, k=4):
    live = set()
    graph = {}

    for line in reversed(ssa):
        if "=" in line:
            lhs, rhs = map(str.strip, line.split("="))
            graph.setdefault(lhs, set())
            for v in live:
                graph[lhs].add(v)
                graph.setdefault(v, set()).add(lhs)
            live.discard(lhs)
            for tok in rhs.split():
                if "_" in tok:
                    live.add(tok)

    registers = {}
    for var in graph:
        used = {registers[n] for n in graph[var] if n in registers}
        for r in range(k):
            if r not in used:
                registers[var] = f"R{r}"
                break
        else:
            registers[var] = "SPILL"

    return registers



def ssa_to_bytecode(ssa, regmap):
    bc = []

    for line in ssa:
        if line.startswith("return"):
            v = line.split()[1]
            bc.append(("LOAD_FAST", regmap[v]))
            bc.append(("RETURN_VALUE", None))

        elif "=" in line:
            lhs, rhs = map(str.strip, line.split("="))
            tokens = rhs.split()
            if "+" in tokens:
                a, _, b = tokens
                bc.append(("LOAD_FAST", regmap[a]))
                bc.append(("LOAD_FAST", regmap[b]))
                bc.append(("BINARY_ADD", None))
                bc.append(("STORE_FAST", regmap[lhs]))

    return bc



class MiniJIT:
    def __init__(self):
        self.hot = {}

    def run(self, func, *args):
        key = (func, tuple(type(a) for a in args))
        self.hot[key] = self.hot.get(key, 0) + 1

        if self.hot[key] > 5:
            return self.compile_and_run(func, args)

        return func(*args)

    def compile_and_run(self, func, args):
        ssa = BytecodeSSA().convert(func)
        ssa = cse(dce(ssa))
        regs = register_allocate(ssa)
        bc = ssa_to_bytecode(ssa, regs)
        return func(*args)  # placeholder



class IR:
    pass

class Add(IR):
    def __init__(self, a, b): self.a, self.b = a, b

class Assign(IR):
    def __init__(self, dst, val): self.dst, self.val = dst, val

class Return(IR):
    def __init__(self, val): self.val = val



x1 = IR.Assign("x1", IR.Add("x0", 1))
x2 = IR.Assign("x2", IR.Add("x0", 1))
x3 = IR.Assign("x3", IR.Add("x2", 2))
ret = IR.Return("x3")



class IR: pass

class Assign(IR):
    def __init__(self, dst, expr): self.dst, self.expr = dst, expr

class Add(IR):
    def __init__(self, a, b): self.a, self.b = a, b

class Return(IR):
    def __init__(self, v): self.v = v



def ir_to_c(ir_list):
    lines = ["int func(int x) {"]

    for ir in ir_list:
        if isinstance(ir, Assign):
            if isinstance(ir.expr, Add):
                lines.append(
                    f"  int {ir.dst} = {ir.expr.a} + {ir.expr.b};"
                )

        elif isinstance(ir, Return):
            lines.append(f"  return {ir.v};")

    lines.append("}")
    return "\n".join(lines)



def ir_to_wasm(ir_list):
    wasm = [
        "(module",
        "  (func $f (param $x i32) (result i32)"
    ]

    for ir in ir_list:
        if isinstance(ir, Assign) and isinstance(ir.expr, Add):
            wasm += [
                f"    local.get ${ir.expr.a}",
                f"    i32.const {ir.expr.b}",
                "    i32.add",
                f"    local.set ${ir.dst}"
            ]

        elif isinstance(ir, Return):
            wasm.append(f"    local.get ${ir.v}")

    wasm += ["  )", ")"]
    return "\n".join(wasm)



import subprocess, ctypes, tempfile, os

def jit_compile(ir):
    c_code = ir_to_c(ir)

    with tempfile.NamedTemporaryFile(suffix=".c", delete=False) as f:
        f.write(c_code.encode())
        cfile = f.name

    sofile = cfile.replace(".c", ".so")

    subprocess.run(
        ["clang", "-shared", "-O3", "-fPIC", cfile, "-o", sofile],
        check=True
    )

    lib = ctypes.CDLL(sofile)
    return lib.func



# if x > 0:
#     y = 1
# else:
#     y = 2
# return y



# from z3 import *

# x = Int("x")
# y1 = Int("y1")
# y2 = Int("y2")

# s = Solver()
# s.add(Implies(x > 0, y1 == 1))
# s.add(Implies(x <= 0, y2 == 2))



# import time

# def bench(f, n=1_000_000):
#     t = time.time()
#     for _ in range(n):
#         f(10)
#     return time.time() - t



# Random-code.py
# Generator file Python superset (fixed & valid)

def main():
    content = []

    # ===== HEADER =====
    content.append("# AUTO-GENERATED PYTHON SUPERSET FILE")
    content.append("# This file intentionally contains many Python constructs\n")

    # ===== ASYNC / AWAIT =====
    content.append("# === async / await ===")
    content.append(
        """
async def tugas_async():
    await tugas_async_kecil()
    return True

async def tugas_async_kecil():
    return False
"""
    )

    # ===== YIELD =====
    content.append("# === yield ===")
    content.append(
        """
def generator():
    yield 1
    yield 2
    yield 3
"""
    )

    # ===== TRY / EXCEPT =====
    content.append("# === try / except ===")
    content.append(
        """
try:
    x = 1 / 1
except ZeroDivisionError:
    x = 0
finally:
    pass
"""
    )

    # ===== PLACEHOLDER / NON-EXECUTED =====
    content.append("# === non-executed / illustrative ===")
    content.append(
        """
# The following are illustrative only:
# break
# continue
# await outside async
# return outside function
"""
    )

    # ===== WRITE OUTPUT =====
    output_file = "python_superset_generated.py"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(content))

    print(f"[OK] File generated: {output_file}")


if __name__ == "__main__":
    main()



import keyword
from datetime import datetime

OUTPUT_FILE = "python_keyword_superset.py"


def generate():
    kw = sorted(keyword.kwlist)
    lines = []

    # =========================================================
    # HEADER
    # =========================================================
    lines.append("# =========================================================")
    lines.append("# PYTHON KEYWORD SUPERSET (COMPLETE & PERFECT)")
    lines.append("# Auto-generated")
    lines.append(f"# Generated at: {datetime.now().isoformat()}")
    lines.append("# =========================================================\n")

    # =========================================================
    # SECTION 1: GLOBAL EXECUTABLE KEYWORDS
    # =========================================================
    lines.append("# === SECTION 1: GLOBAL EXECUTABLE KEYWORDS ===\n")
    lines.append("a = True")
    lines.append("b = False")
    lines.append("c = None\n")

    lines.append("if a and not b or c is None:")
    lines.append("    pass\n")

    lines.append("x = 1 in [1, 2, 3]\n")

    # =========================================================
    # SECTION 2: FUNCTION CONTEXT KEYWORDS
    # =========================================================
    lines.append("# === SECTION 2: FUNCTION CONTEXT KEYWORDS ===\n")

    lines.append("def fungsi_dasar():")
    lines.append("    global g")
    lines.append("    g = 10")
    lines.append("    return g\n")

    lines.append("def fungsi_nonlocal():")
    lines.append("    x = 1")
    lines.append("    def inner():")
    lines.append("        nonlocal x")
    lines.append("        x += 1")
    lines.append("        return x")
    lines.append("    return inner()\n")

    lines.append("f_lambda = lambda x: x + 1\n")

    # =========================================================
    # SECTION 3: LOOP CONTEXT KEYWORDS
    # =========================================================
    lines.append("# === SECTION 3: LOOP CONTEXT KEYWORDS ===\n")

    lines.append("for i in range(3):")
    lines.append("    if i == 1:")
    lines.append("        continue")
    lines.append("    if i == 2:")
    lines.append("        break\n")

    lines.append("while False:")
    lines.append("    pass\n")

    # =========================================================
    # SECTION 4: EXCEPTION HANDLING
    # =========================================================
    lines.append("# === SECTION 4: EXCEPTION HANDLING ===\n")

    lines.append("try:")
    lines.append("    assert True")
    lines.append("except Exception as e:")
    lines.append("    err = e")
    lines.append("finally:")
    lines.append("    pass\n")

    # =========================================================
    # SECTION 5: CLASS & OOP
    # =========================================================
    lines.append("# === SECTION 5: CLASS & OOP ===\n")

    lines.append("class ContohClass:")
    lines.append("    def __init__(self, nilai):")
    lines.append("        self.nilai = nilai")
    lines.append("")
    lines.append("    def get(self):")
    lines.append("        return self.nilai\n")

    # =========================================================
    # SECTION 6: GENERATOR & YIELD
    # =========================================================
    lines.append("# === SECTION 6: GENERATOR & YIELD ===\n")

    lines.append("def generator_angka():")
    lines.append("    yield 1")
    lines.append("    yield 2")
    lines.append("    yield 3\n")

    # =========================================================
    # SECTION 7: ASYNC / AWAIT
    # =========================================================
    lines.append("# === SECTION 7: ASYNC / AWAIT ===\n")

    lines.append("async def async_kecil():")
    lines.append("    return True\n")

    lines.append("async def async_besar():")
    lines.append("    hasil = await async_kecil()")
    lines.append("    return hasil\n")

    # =========================================================
    # SECTION 8: PATTERN MATCHING
    # =========================================================
    lines.append("# === SECTION 8: PATTERN MATCHING ===\n")

    lines.append("def cocokkan(x):")
    lines.append("    match x:")
    lines.append("        case 1:")
    lines.append("            return 'satu'")
    lines.append("        case _:")
    lines.append("            return 'lainnya'\n")

    # =========================================================
    # SECTION 9: IMPORT SYSTEM
    # =========================================================
    lines.append("# === SECTION 9: IMPORT SYSTEM ===\n")

    lines.append("import math as m")
    lines.append("from math import sqrt as akar\n")

    # =========================================================
    # SECTION 10: DELETION
    # =========================================================
    lines.append("# === SECTION 10: DELETION ===\n")

    lines.append("temp = 123")
    lines.append("del temp\n")

    # =========================================================
    # SECTION 11: GRAMMAR-VALID BUT NON-EXECUTABLE
    # =========================================================
    lines.append("# === SECTION 11: GRAMMAR-VALID BUT NON-EXECUTABLE ===\n")

    lines.append('"""')
    lines.append("return 1            # illegal at global scope")
    lines.append("break               # illegal outside loop")
    lines.append("continue            # illegal outside loop")
    lines.append("await x             # illegal outside async")
    lines.append('"""\n')

    # =========================================================
    # SECTION 12: KEYWORD LIST (CANONICAL)
    # =========================================================
    lines.append("# === SECTION 12: CANONICAL KEYWORD LIST ===\n")
    lines.append("PYTHON_KEYWORDS = [")
    for k in kw:
        lines.append(f"    '{k}',")
    lines.append("]\n")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Superset generated: {OUTPUT_FILE}")
    print(f"[INFO] Total keywords: {len(kw)}")


if __name__ == "__main__":
    generate()



# =========================================================
# PYTHON KEYWORD SUPERSET (COMPLETE & PERFECT)
# Auto-generated
# =========================================================

# === SECTION 1: GLOBAL EXECUTABLE KEYWORDS ===
a = True
b = False
c = None

if a and not b or c is None:
    pass

x = 1 in [1, 2, 3]

# === SECTION 2: FUNCTION CONTEXT KEYWORDS ===
def fungsi_dasar():
    global g
    g = 10
    return g

def fungsi_nonlocal():
    x = 1
    def inner():
        nonlocal x
        x += 1
        return x
    return inner()

f_lambda = lambda x: x + 1

# === SECTION 3: LOOP CONTEXT KEYWORDS ===
for i in range(3):
    if i == 1:
        continue
    if i == 2:
        break

while False:
    pass

# === SECTION 4: EXCEPTION HANDLING ===
try:
    assert True
except Exception as e:
    err = e
finally:
    pass

# === SECTION 11: GRAMMAR-VALID BUT NON-EXECUTABLE ===
"""
return 1
break
continue
await x
"""

# === SECTION 12: CANONICAL KEYWORD LIST ===
PYTHON_KEYWORDS = [
    'False',
    'None',
    'True',
    'and',
    'as',
    'assert',
    'async',
    'await',
    ...
]
