import imutils
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk  # Nhập Pillow để hiển thị hình ảnh


# Hàm chọn hình ảnh
def select_image():
    file_path = filedialog.askopenfilename(title="Chọn một hình ảnh",
                                           filetypes=(("Tệp PNG", "*.png"),
                                                      ("Tệp JPEG", "*.jpg;*.jpeg"),
                                                      ("Tất cả tệp", "*.*")))
    if file_path:
        return file_path
    return None


# Hàm tạo các trường nhập đáp án
def create_answer_entries(num_questions):
    # Xóa các trường nhập đáp án hiện có nếu có
    for widget in answer_frame.winfo_children():
        widget.destroy()

    # Tạo các trường nhập cho mỗi câu hỏi
    for i in range(num_questions):
        label = tk.Label(answer_frame, text=f"Câu {i + 1}:")
        label.pack(pady=5)

        entry = tk.Entry(answer_frame)
        entry.pack(pady=5)
        answer_entries.append(entry)


# Hàm để nhập các đáp án đúng
def get_answer_key():
    answer_key = {}
    for i, entry in enumerate(answer_entries):
        answer = entry.get().strip().upper()
        if answer and answer in ["A", "B", "C", "D", "E"]:
            answer_key[i] = ord(answer) - ord('A')  # Chuyển đổi đáp án chữ cái thành số
        else:
            messagebox.showerror("Lỗi", f"Đáp án không hợp lệ cho câu hỏi {i + 1}. Vui lòng nhập A, B, C, D hoặc E.")
            return None
    return answer_key


# Hàm xử lý OMR
def process_omr(answer_key, test_image_path):
    # Tải hình ảnh kiểm tra
    test_image = cv2.imread(test_image_path)

    # Chuyển đổi sang ảnh xám và xử lý
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # Tìm các đường viền trong hình ảnh đã được xử lý
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    docCnt = None

    # Đảm bảo ít nhất một đường viền đã được tìm thấy
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break

    if docCnt is None:
        messagebox.showerror("Lỗi", "Không tìm thấy đường viền tài liệu.")
        return

    # Áp dụng biến đổi phối cảnh bốn điểm
    paper = four_point_transform(test_image, docCnt.reshape(4, 2))
    warped = four_point_transform(gray, docCnt.reshape(4, 2))

    # Áp dụng ngưỡng Otsu
    thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Tìm các đường viền trong hình ảnh đã ngưỡng
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    questionCnts = []

    # Lặp qua các đường viền
    for c in cnts:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
            questionCnts.append(c)

    # Sắp xếp các đường viền câu hỏi từ trên xuống dưới
    questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
    correct = 0

    # Mỗi câu hỏi có 5 đáp án có thể
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
        cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
        bubbled = None

        for (j, c) in enumerate(cnts):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)

            if bubbled is None or total > bubbled[0]:
                bubbled = (total, j)

        color = (0, 0, 255)
        k = answer_key[q]

        if k == bubbled[1]:
            color = (0, 255, 0)
            correct += 1

        cv2.drawContours(paper, [cnts[k]], -1, color, 3)

    score = (correct / len(answer_key)) * 100
    messagebox.showinfo("Điểm", f"Điểm: {score:.2f}%")
    cv2.putText(paper, "{:.2f}%".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Hiển thị các hình ảnh
    cv2.imshow("Hình gốc", test_image)
    cv2.imshow("Bài kiểm tra", paper)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Tạo cửa sổ Tkinter
root = tk.Tk()
root.title("Công cụ OMR")

# Thêm nút chọn hình ảnh kiểm tra
test_image_path = None
answer_entries = []  # Danh sách để giữ các widget nhập đáp án
answer_frame = tk.Frame(root)  # Khung cho các trường nhập đáp án
answer_frame.pack(pady=10)


def load_test_image():
    global test_image_path
    test_image_path = select_image()
    if test_image_path:
        display_image(test_image_path)
        test_button.config(text="Hình ảnh kiểm tra đã được chọn")
        create_answer_entries(5)  # Thay 5 bằng số lượng câu hỏi thực tế


def display_image(image_path):
    # Hiển thị hình ảnh đã chọn trong một cửa sổ mới
    img = Image.open(image_path)
    img = img.resize((400, 400), Image.LANCZOS)  # Thay đổi kích thước để hiển thị
    img_tk = ImageTk.PhotoImage(img)

    image_window = tk.Toplevel(root)
    image_window.title("Hình ảnh đã chọn")
    label = tk.Label(image_window, image=img_tk)
    label.image = img_tk  # Giữ tham chiếu để tránh thu gom rác
    label.pack()


test_button = tk.Button(root, text="Chọn hình ảnh kiểm tra", command=load_test_image)
test_button.pack(pady=10)


# Thêm nút để nhập đáp án và xử lý OMR
def start_omr_process():
    if test_image_path:
        answer_key = get_answer_key()
        if answer_key:  # Đảm bảo rằng khóa đáp án không rỗng
            process_omr(answer_key, test_image_path)
        else:
            messagebox.showerror("Lỗi", "Vui lòng nhập đáp án hợp lệ.")
    else:
        messagebox.showerror("Lỗi", "Vui lòng chọn một hình ảnh kiểm tra trước.")


process_button = tk.Button(root, text="Nhập đáp án và chấm điểm", command=start_omr_process)
process_button.pack(pady=20)

# Chạy vòng lặp chính
root.mainloop()
