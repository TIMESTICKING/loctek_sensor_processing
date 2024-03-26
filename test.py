import curses

def main(stdscr):
    # 关闭屏幕回显
    curses.noecho()
    # 响应键盘输入，而无需按回车
    curses.cbreak()
    # 隐藏光标
    curses.curs_set(0)
    # 设置窗口接受键盘输入
    stdscr.keypad(True)

    stdscr.addstr("按 'q' 键退出程序\n")
    while True:
        # 非阻塞地获取按键输入
        key = stdscr.getch()
        if key == ord('q'):
            break
        elif key != -1:  # 如果有按键被按下
            stdscr.addstr(f"检测到按键: {chr(key)}\n")

    # 恢复终端设置
    curses.nocbreak()
    stdscr.keypad(False)
    curses.echo()
    curses.endwin()

if __name__ == '__main__':
    curses.wrapper(main)
