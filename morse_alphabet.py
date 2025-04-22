CHARS = sorted(list("АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789 "))
# blank токен для CTC – всегда в позиции 0
IDX_TO_CHAR = ["<blank>"] + CHARS
CHAR_TO_IDX = {ch: i + 1 for i, ch in enumerate(CHARS)}
