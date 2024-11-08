from Crypto.Util.number import getPrime as gp, inverse as inv, long_to_bytes as l2b, bytes_to_long as b2l
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
from moviepy.editor import VideoFileClip, VideoClip
from moviepy.audio.AudioClip import AudioArrayClip
import pillow_heif
from PIL import Image
import io
import threading
import time
import sys
import math
import os
import cv2
import numpy as np
from typing import Tuple, List, Optional, Union

logo = """ 
          .         .                                                                              
         ,8.       ,8.           ,o888888o.         ,o888888o.     8 8888888888 `8.`8888.      ,8' 
        ,888.     ,888.       . 8888     `88.    . 8888     `88.   8 8888        `8.`8888.    ,8'  
       .`8888.   .`8888.     ,8 8888       `8b  ,8 8888       `8b  8 8888         `8.`8888.  ,8'   
      ,8.`8888. ,8.`8888.    88 8888        `8b 88 8888        `8b 8 8888          `8.`8888.,8'    
     ,8'8.`8888,8^8.`8888.   88 8888         88 88 8888         88 8 888888888888   `8.`88888'     
    ,8' `8.`8888' `8.`8888.  88 8888         88 88 8888         88 8 8888            `8. 8888      
   ,8'   `8.`88'   `8.`8888. 88 8888        ,8P 88 8888        ,8P 8 8888             `8 8888      
  ,8'     `8.`'     `8.`8888.`8 8888       ,8P  `8 8888       ,8P  8 8888              8 8888      
 ,8'       `8        `8.`8888.` 8888     ,88'    ` 8888     ,88'   8 8888              8 8888      
,8'         `         `8.`8888.  `8888888P'         `8888888P'     8 8888              8 8888      
"""

RESET = '\033[0m'
CLEAR_AND_RETURN = '\033[H'
SAVE_CURSOR = '\033[s'
RESTORE_CURSOR = '\033[u'
stop_animation = False
output = None

class OutputManager:
    def __init__(self):
        self.current_line = 0
        self.logo_height = len(logo.split('\n'))
        self.start_position = self.logo_height + 2

    def print(self, text, end='\n'):
        move_cursor(0, self.start_position + self.current_line)
        print('\033[K', end='')
        print(text, end=end)
        if end == '\n':
            self.current_line += 1

    def reset(self):
        self.current_line = 0
        for i in range(20):
            move_cursor(0, self.start_position + i)
            print('\033[K', end='')
        move_cursor(0, self.start_position)

    def input(self, prompt):
        move_cursor(0, self.start_position + self.current_line)
        print('\033[K', end='')
        result = input(prompt)
        self.current_line += 1
        return result

def get_terminal_size():
    return os.get_terminal_size()

def clear_screen():
    print('\033[2J', end='')
    print('\033[H', end='')

def move_cursor(x, y):
    print(f'\033[{y};{x}H', end='')
    
def rgb_to_ansi(r, g, b):
    return f'\033[38;2;{r};{g};{b}m'

def animate_logo():
    global stop_animation
    while not stop_animation:
        for i in range(360):
            if stop_animation:
                break
            
            h = i / 360
            s = 1.0
            v = 1.0
            
            hi = int(h * 6)
            f = h * 6 - hi
            p = v * (1 - s)
            q = v * (1 - f * s)
            t = v * (1 - (1 - f) * s)

            if hi == 0:
                r, g, b = v, t, p
            elif hi == 1:
                r, g, b = q, v, p
            elif hi == 2:
                r, g, b = p, v, t
            elif hi == 3:
                r, g, b = p, q, v
            elif hi == 4:
                r, g, b = t, p, v
            else:
                r, g, b = v, p, q

            r = int(r * 255)
            g = int(g * 255)
            b = int(b * 255)

            print(SAVE_CURSOR, end='')
            move_cursor(0, 0)
            print(f"{rgb_to_ansi(r,g,b)}{logo}{RESET}", end='')
            print(RESTORE_CURSOR, end='')
            sys.stdout.flush()
            time.sleep(0.05)

def get_max_block_size(n):
    return math.floor(math.log(n, 256))

def get_file_type(filename: str) -> str:
    video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.heic']
    encrypted_formats = ['.bin', '.txt']
    
    ext = os.path.splitext(filename)[1].lower()
    base_name = os.path.splitext(filename)[0]
    
    if ext in encrypted_formats:
        params_file = f"params_for_{base_name}.txt"
        if os.path.exists(params_file):
            return 'video'
        return 'image'
    
    if ext in video_formats:
        return 'video'
    elif ext in image_formats:
        return 'image'
    else:
        return 'unknown'


def save_video_params(filename: str, params: dict) -> None:
    try:
        params_filename = f"params_for_{os.path.splitext(filename)[0]}.txt"
        with open(params_filename, 'w') as f:
            for key, value in params.items():
                f.write(f"{key}:{value}\n")
        output.print(f"Параметры видео сохранены в файле {params_filename}")
    except Exception as e:
        output.print(f"Ошибка при сохранении параметров видео: {e}")

def read_video_params(filename: str) -> Optional[dict]:
    try:
        params_filename = f"params_for_{os.path.splitext(filename)[0]}.txt"
        params = {}
        with open(params_filename, 'r') as f:
            for line in f:
                key, value = line.strip().split(':')
                params[key] = value
        return params
    except Exception as e:
        output.print(f"Ошибка при чтении параметров видео: {e}")
        return None

def process_video_for_encryption(filename: str) -> Tuple[bytes, dict]:
    try:
        cap = cv2.VideoCapture(filename)
        if not cap.isOpened():
            raise Exception("Не удалось открыть видеофайл")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output.print(f"Обработка видео: {width}x{height} @ {fps}fps")
        
        frame_size = width * height * 3
        frames_bytes = bytearray(frame_size * total_frames)
        offset = 0
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_bytes = frame.tobytes()
            frames_bytes[offset:offset + frame_size] = frame_bytes
            offset += frame_size
            frame_count += 1
            
            if frame_count % 10 == 0:
                output.print(f"Обработано кадров: {frame_count}/{total_frames}", end='\r')

        cap.release()

        frames_bytes = frames_bytes[:offset]
        
        video_params = {
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'frame_size': frame_size
        }
        
        return bytes(frames_bytes), video_params

    except Exception as e:
        output.print(f"Ошибка при обработке видео: {e}")
        return None, None

def save_decrypted_video(filename: str, data: bytes, params: dict) -> None:
    try:
        width = int(params['width'])
        height = int(params['height'])
        fps = int(params['fps'])
        frame_count = int(params['frame_count'])
        frame_size = int(params['frame_size'])

        all_frames = np.frombuffer(data[:frame_count * frame_size], dtype=np.uint8)
        all_frames = all_frames.reshape(frame_count, height, width, 3)

        output_path = f"{os.path.splitext(filename)[0]}_decrypted{os.path.splitext(filename)[1]}"
        
        out = cv2.VideoWriter(output_path, 
                            cv2.VideoWriter_fourcc(*'XVID'), 
                            fps, 
                            (width, height))

        batch_size = 50
        for i in range(0, frame_count, batch_size):
            batch_end = min(i + batch_size, frame_count)
            batch_frames = all_frames[i:batch_end]
            
            for frame in batch_frames:
                out.write(frame)
            
            if i % (batch_size * 2) == 0:
                output.print(f"Сохранено кадров: {batch_end}/{frame_count}", end='\r')

        out.release()
        output.print(f"\nВидео сохранено в {output_path}")

    except Exception as e:
        output.print(f"Ошибка при сохранении видео: {e}")



def encrypt_video_aes(filename: str) -> None:
    try:
        video = VideoFileClip(filename)
        
        width = int(video.size[0])
        height = int(video.size[1])
        fps = video.fps
        duration = video.duration
        
        output.print(f"Параметры видео: {width}x{height} @ {fps}fps")
        output.print(f"Длительность: {duration} секунд")

        # Сохраняем аудио отдельно
        has_audio = video.audio is not None
        if has_audio:
            try:
                # Преобразуем аудио в массив и сохраняем все его параметры
                audio_data = video.audio.to_soundarray()
                audio_fps = video.audio.fps
                audio_duration = video.audio.duration
                # Если аудио стерео, сохраняем все каналы
                audio_channels = audio_data.shape[1] if len(audio_data.shape) > 1 else 1
                output.print(f"Аудио: {audio_fps}Hz, {audio_channels} каналов")
            except Exception as e:
                output.print(f"Предупреждение: Не удалось получить аудио: {e}")
                has_audio = False
                audio_data = None
                audio_fps = None
                audio_duration = None
                audio_channels = None
        else:
            audio_data = None
            audio_fps = None
            audio_duration = None
            audio_channels = None

        frames = []
        total_frames = int(duration * fps)
        progress = 0
        for frame in video.iter_frames():
            frames.append(frame.tobytes())
            progress += 1
            if progress % 10 == 0:
                output.print(f"Обработано кадров: {progress}/{total_frames}", end='\r')

        video.close()

        video_data = {
            'width': width,
            'height': height,
            'fps': fps,
            'frames': frames,
            'has_audio': has_audio,
            'audio_data': audio_data,
            'audio_fps': audio_fps,
            'audio_duration': audio_duration,
            'audio_channels': audio_channels,
            'duration': duration
        }
        
        import pickle
        serialized_data = pickle.dumps(video_data, protocol=4)  # Используем protocol 4 для больших данных
        
        output.print("\nГенерация AES ключа...")
        aes_key = get_random_bytes(32)
        cipher = AES.new(aes_key, AES.MODE_CBC)
        
        output.print("Начало шифрования данных...")
        padded_data = pad(serialized_data, AES.block_size)
        encrypted_data = cipher.encrypt(padded_data)

        output.print("Генерация RSA ключей...")
        p = gp(1024)
        q = gp(1024)
        n = p * q
        phi = (p - 1) * (q - 1)
        e = 65537
        
        encrypted_aes_key = pow(int.from_bytes(aes_key, 'big'), e, n)

        enc_filename = f"{os.path.splitext(filename)[0]}_encrypted.bin"
        with open(enc_filename, 'wb') as f:
            f.write(cipher.iv)
            f.write(len(encrypted_data).to_bytes(8, 'big'))
            f.write(encrypted_data)

        key_data = f'p:{p}\nn:{n}\ne:{e}\nencrypted_aes_key:{encrypted_aes_key}'
        save_key(filename, key_data)
        
        output.print("Шифрование завершено успешно")

    except Exception as e:
        output.print(f"Произошла ошибка при шифровании: {e}")
        import traceback
        output.print(traceback.format_exc())

def decrypt_video_aes(filename: str, keys: dict = None) -> None:
    try:
        if keys is None:
            key_filename = output.input("Введите имя файла с ключом: ")
            keys = read_key_file(key_filename)
            if keys is None:
                return

        base_filename = filename.replace('_encrypted', '')
        enc_filename = f"{os.path.splitext(base_filename)[0]}_encrypted.bin"
        
        output.print("Чтение заголовка файла...")
        with open(enc_filename, 'rb') as f:
            iv = f.read(16)
            data_size = int.from_bytes(f.read(8), 'big')
            output.print(f"Размер зашифрованных данных: {data_size/1024/1024:.2f} MB")

        # Восстанавливаем AES ключ
        p = keys['p']
        n = keys['n']
        encrypted_aes_key = keys['encrypted_aes_key']
        
        phi = (p - 1) * (n//p - 1)
        d = inv(65537, phi)
        
        aes_key = pow(encrypted_aes_key, d, n).to_bytes(32, 'big')
        cipher = AES.new(aes_key, AES.MODE_CBC, iv)

        output.print("Начало расшифрования...")
        
        # Создаем временный файл для расшифрованных данных
        temp_decrypted = "temp_decrypted.bin"
        CHUNK_SIZE = 64 * 1024 * 1024  # 64MB chunks
        
        decrypted_size = 0
        with open(enc_filename, 'rb') as inf, open(temp_decrypted, 'wb') as outf:
            # Пропускаем заголовок
            inf.seek(24)  # 16 байт IV + 8 байт размера
            
            while decrypted_size < data_size:
                chunk = inf.read(min(CHUNK_SIZE, data_size - decrypted_size))
                if not chunk:
                    break
                decrypted_chunk = cipher.decrypt(chunk)
                outf.write(decrypted_chunk)
                decrypted_size += len(chunk)
                output.print(f"Расшифровано: {decrypted_size*100/data_size:.1f}%", end='\r')

        output.print("\nПрименение unpadding...")
        # Читаем последний блок для unpadding
        with open(temp_decrypted, 'rb+') as f:
            f.seek(-AES.block_size, 2)
            last_block = f.read()
            unpadded_block = unpad(last_block, AES.block_size)
            f.seek(-AES.block_size, 2)
            f.write(unpadded_block)
            f.truncate()

        output.print("Загрузка данных видео...")
        with open(temp_decrypted, 'rb') as f:
            import pickle
            video_data = pickle.loads(f.read())

        os.remove(temp_decrypted)
        output.print("Создание видеофайла...")

        # Создаем временные файлы
        temp_video = "temp_video_no_audio.mp4"
        temp_audio = "temp_audio.mp3"

        def make_frame(t):
            frame_idx = int(t * video_data['fps'])
            if frame_idx >= len(video_data['frames']):
                frame_idx = len(video_data['frames']) - 1
            frame = np.frombuffer(video_data['frames'][frame_idx], dtype=np.uint8)
            return frame.reshape((video_data['height'], video_data['width'], 3))

        video_clip = VideoClip(make_frame)
        video_clip = video_clip.set_duration(video_data['duration'])
        video_clip = video_clip.set_fps(video_data['fps'])

        output.print("Сохранение видео...")
        output_filename = output.input("Введите имя файла для сохранения расшифрованного файла: ")

        # Сохраняем видео напрямую с оптимизированными параметрами
        video_clip.write_videofile(
            temp_video,
            codec='libx264',
            fps=video_data['fps'],
            threads=4,
            preset='ultrafast',
            audio=False,
            logger=None
        )

        if video_data['has_audio'] and video_data['audio_data'] is not None:
            try:
                output.print("Обработка аудио...")
                audio_array = np.array(video_data['audio_data'])
                audio_clip = AudioArrayClip(audio_array, fps=video_data['audio_fps'])
                audio_clip.write_audiofile(temp_audio, logger=None)

                output.print("Объединение видео и аудио...")
                # Используем ffmpeg напрямую для объединения с минимальной обработкой
                os.system(f'ffmpeg -v quiet -stats -i {temp_video} -i {temp_audio} -c:v copy -c:a aac -strict experimental "{output_filename}"')

                os.remove(temp_video)
                os.remove(temp_audio)
            except Exception as e:
                output.print(f"Ошибка при обработке аудио: {e}")
                os.rename(temp_video, output_filename)
        else:
            os.rename(temp_video, output_filename)

        video_clip.close()
        output.print("Расшифрование завершено успешно")

    except Exception as e:
        output.print(f"Произошла ошибка при расшифровке: {e}")
        import traceback
        output.print(traceback.format_exc())
        # Очистка временных файлов при ошибке
        for temp_file in [temp_decrypted, temp_video, temp_audio]:
            if os.path.exists(temp_file):
                os.remove(temp_file)


def get_image_bytes(filename: str) -> Optional[bytes]:
    try:
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.heic':
            heif_file = pillow_heif.read_heif(filename)
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
            with io.BytesIO() as bio:
                image.save(bio, format='PNG')
                return bio.getvalue()
        else:
            with open(filename, 'rb') as f:
                return f.read()
    except FileNotFoundError:
        output.print(f"Ошибка: Файл {filename} не найден")
        return None
    except Exception as e:
        output.print(f"Ошибка при чтении файла: {e}")
        return None


def save_encrypted_data(filename: str, data: List[str], original_size: int) -> None:
    try:
        enc_filename = f"{os.path.splitext(filename)[0]}_encrypted.txt"
        
        with open(enc_filename, 'w', buffering=8*1024*1024) as f:
            f.write(str(original_size) + '\n')
            batch_size = 1000
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                f.write('\n'.join(batch) + '\n')
            
        output.print(f"Зашифрованные данные сохранены в файле {enc_filename}")
    except Exception as e:
        output.print(f"Ошибка при сохранении зашифрованных данных: {e}")

def read_key_file(filename):
    try:
        with open(filename, 'r') as f:
            content = f.read()
            key_dict = {}
            for line in content.split('\n'):
                if ':' in line:
                    key, value = line.split(':')
                    key_dict[key] = int(value)
            return key_dict
    except FileNotFoundError:
        output.print(f"Ошибка: Файл ключа {filename} не найден")
        return None
    except Exception as e:
        output.print(f"Ошибка при чтении файла ключа: {e}")
        return None

def read_encrypted_data(filename: str, file_type: str) -> Tuple[Union[List[str], bytes], int]:
    try:
        base_name = filename.replace('_encrypted', '')  # Убираем _encrypted если оно есть
        if file_type == 'video':
            enc_filename = f"{os.path.splitext(base_name)[0]}_encrypted.bin"
            with open(enc_filename, 'rb') as f:
                iv = f.read(16)  # Читаем IV
                data_size = int.from_bytes(f.read(8), 'big')  # Читаем размер данных
                encrypted_data = f.read(data_size)
                return (iv, encrypted_data), data_size
        else:
            enc_filename = f"{os.path.splitext(base_name)[0]}_encrypted.txt"
            encrypted_blocks = []
            with open(enc_filename, 'r', buffering=8*1024*1024) as f:
                original_size = int(f.readline().strip())
                while True:
                    lines = f.readlines(1024*1024)
                    if not lines:
                        break
                    encrypted_blocks.extend(line.strip() for line in lines if line.strip())
                return encrypted_blocks, original_size
                
    except FileNotFoundError:
        output.print(f"Ошибка: Файл {enc_filename} не найден")
        return None, None
    except Exception as e:
        output.print(f"Ошибка при чтении зашифрованных данных: {e}")
        return None, None
                
    except FileNotFoundError:
        output.print(f"Ошибка: Файл {filename} не найден")
        return None, None
    except Exception as e:
        output.print(f"Ошибка при чтении зашифрованных данных: {e}")
        return None, None

def encrypt_block(block, e, n):
    m = b2l(block)
    if m >= n:
        raise ValueError(f"Блок слишком большой: {m} >= {n}")
    c = pow(m, e, n)
    return str(c)

def decrypt_block(block_str, d, n, block_size):
    try:
        c = int(block_str)
        m = pow(c, d, n)
        decrypted = l2b(m)
        if len(decrypted) < block_size:
            decrypted = b'\x00' * (block_size - len(decrypted)) + decrypted
        return decrypted
    except Exception as e:
        output.print(f"Ошибка при расшифровке блока: {e}")
        return None

def encrypt_file(filename: str) -> None:
    file_type = get_file_type(filename)
    
    if file_type == 'video':
        encrypt_video_aes(filename)
    elif file_type == 'image':
        output.reset()
        data = get_image_bytes(filename)
        if data is None:
            return

        original_size = len(data)
        output.print(f"Размер исходного файла: {original_size} байт")
        output.print("Генерация ключей...")
        
        p = gp(1024)
        q = gp(1024)
        n = p * q
        phi = (p - 1) * (q - 1)
        e = 65537
        
        block_size = get_max_block_size(n) - 1
        output.print(f"Используемый размер блока: {block_size} байт")

        encrypted_blocks = []
        total_blocks = (len(data) + block_size - 1) // block_size
        
        output.print("Начало шифрования...")
        progress_line = output.current_line - 1
        for i in range(0, len(data), block_size):
            block = data[i:i+block_size]
            if len(block) < block_size:
                block = block.ljust(block_size, b'\x00')
            
            try:
                encrypted_block = encrypt_block(block, e, n)
                encrypted_blocks.append(encrypted_block)
            except ValueError as ve:
                output.print(f"Ошибка при шифровании блока {i//block_size}: {ve}")
                return
            
            current_block = i // block_size + 1
            move_cursor(0, output.start_position + progress_line)
            print('\033[K', end='')
            print(f"Прогресс: {current_block}/{total_blocks} блоков обработано")

        output.print("\nШифрование завершено")
        save_encrypted_data(filename, encrypted_blocks, original_size)
        
        # Сохраняем все необходимые параметры для RSA шифрования изображений
        key = {
            'p': p,
            'n': n,
            'e': e,
            'block_size': block_size
        }
        key_str = '\n'.join(f'{k}:{v}' for k, v in key.items())
        save_key(filename, key_str)
        
        output.print("Шифрование завершено успешно")
    else:
        output.print("Неподдерживаемый формат файла")

def get_encryption_type(keys: dict) -> str:
    """Определяет тип шифрования на основе содержимого ключевого файла"""
    if 'encrypted_aes_key' in keys:
        return 'aes'
    elif 'block_size' in keys:
        return 'rsa'
    else:
        return 'unknown'

def decrypt_file(filename: str) -> None:
    file_type = get_file_type(filename)
    
    output.reset()
    key_filename = output.input("Введите имя файла с ключом: ")
    keys = read_key_file(key_filename)
    if keys is None:
        return

    # Определяем тип шифрования по ключам
    encryption_type = get_encryption_type(keys)
    
    if encryption_type == 'aes':
        decrypt_video_aes(filename, keys)
    elif encryption_type == 'rsa':
        # RSA расшифровка
        encrypted_blocks, original_size = read_encrypted_data(filename, 'image')
        if encrypted_blocks is None:
            return

        output.print(f"Размер исходного файла: {original_size} байт")
        output.print(f"Количество блоков: {len(encrypted_blocks)}")

        required_keys = ['p', 'n', 'e', 'block_size']
        if not all(key in keys for key in required_keys):
            output.print("Ошибка: Файл с ключом имеет неверный формат для RSA")
            return

        p = keys['p']
        n = keys['n']
        e = keys['e']
        block_size = keys['block_size']
        
        q = n // p
        if n % p != 0:
            output.print("Ошибка: Неверные ключи")
            return

        phi = (p - 1) * (q - 1)
        d = inv(e, phi)

        decrypted_data = bytearray()
        total_blocks = len(encrypted_blocks)
        
        output.print("Начало расшифрования...")
        progress_line = output.current_line - 1
        for i, block in enumerate(encrypted_blocks):
            decrypted_block = decrypt_block(block, d, n, block_size)
            if decrypted_block is None:
                output.print(f"\nОшибка при расшифровке блока {i}")
                continue
                
            decrypted_data.extend(decrypted_block)
            move_cursor(0, output.start_position + progress_line)
            print('\033[K', end='')
            print(f"Прогресс: {i+1}/{total_blocks} блоков обработано")

        output.print("\nРасшифрование завершено")
        
        output_filename = output.input("Введите имя файла для сохранения расшифрованного файла: ")
        save_image(output_filename, decrypted_data, original_size)
        
        output.print("Расшифрование завершено успешно")
    else:
        output.print("Неизвестный тип шифрования")

def save_image(filename: str, data: bytes, original_size: int):
    try:
        ext = os.path.splitext(filename)[1].lower()
        data = data[:original_size]
        
        if ext == '.heic':
            output.print("HEIC формат не поддерживается для сохранения, конвертируем в PNG")
            filename = f"{os.path.splitext(filename)[0]}.png"
        
        with open(filename, 'wb') as f:
            f.write(data)
        output.print(f"Файл успешно сохранен как {filename}")
    except Exception as e:
        output.print(f"Ошибка при сохранении файла: {e}")

def save_key(filename: str, key: str) -> bool:
    try:
        if not filename or not key:
            output.print("Ошибка: пустое имя файла или ключ")
            return False

        base_name = os.path.splitext(filename)[0]
        safe_name = ''.join(c for c in base_name if c.isalnum() or c in ('_', '-', '.'))
        key_filename = f"key_for_{safe_name}.txt"

        directory = os.path.dirname(key_filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        output.print(f"Попытка сохранения ключа в файл {key_filename}")
        
        with open(key_filename, 'w') as f:
            f.write(key)
        
        if os.path.exists(key_filename):
            output.print(f"Ключ успешно сохранен в файле {key_filename}")
            return True
        else:
            output.print(f"Ошибка: файл ключа не был создан")
            return False
            
    except PermissionError:
        output.print(f"Ошибка: нет прав на создание файла {key_filename}")
        return False
    except OSError as e:
        output.print(f"Ошибка операционной системы при сохранении ключа: {e}")
        return False
    except Exception as e:
        output.print(f"Непредвиденная ошибка при сохранении ключа: {e}")
        import traceback
        output.print(traceback.format_exc())
        return False

def main():
    global stop_animation, output
    
    output = OutputManager()
    clear_screen()
    print('\n' * (len(logo.split('\n')) + 2))

    animation_thread = threading.Thread(target=animate_logo)
    animation_thread.daemon = True
    animation_thread.start()

    while True:
        try:
            output.reset()
            filename = output.input("Введите имя файла: ")
            action = output.input("Выберите действие (1 - зашифровать, 2 - расшифровать): ")

            if action == "1":
                encrypt_file(filename)
            elif action == "2":
                decrypt_file(filename)
            else:
                output.print("Неверное действие. Пожалуйста, выберите 1 или 2")

            # Добавляем очистку буфера ввода
            import sys, termios
            termios.tcflush(sys.stdin, termios.TCIOFLUSH)

            cont = output.input("Хотите продолжить? (y/n): ").lower()
            if cont != 'y':
                stop_animation = True
                break

        except Exception as e:
            output.print(f"Произошла ошибка: {e}")
    
    stop_animation = True
    time.sleep(0.1)

if __name__ == "__main__":
    main()
