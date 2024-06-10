import platform
import psutil
import os



osInf = platform.uname()
print(f"Тип системы:", osInf.system)
print(f"Версия системы:", osInf.version)
print(f"Архитектура системы:", osInf.machine)
print(f"Процессор:", osInf.processor)

svmem = psutil.virtual_memory()
convertMd=1024**2
print(f"Всего памяти:", svmem.total//convertMd, "MB")
print(f"Доступной памяти:", svmem.available//convertMd, "MB")

cpu = psutil.cpu_freq()
print(f"Текущая частота работы процессора:", cpu.current, "Mhz")
print("Количество логических ядер процессора:", os.cpu_count())