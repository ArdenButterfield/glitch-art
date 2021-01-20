from glitcher import Glitcher

dir = '../images/raw/'
branches = dir + 'branches.jpg'
t1_png = dir + 'test1.png'

def test_rotation():
    a = Glitcher()
    a.load_image(branches)
    a.rotate(1)
    a.display()

def test_checkpoints():
    a = Glitcher()

    a.load_image(branches)
    a.set_checkpoint('first')

    a.rotate(1)
    a.set_checkpoint()

    a.invert_colors()
    a.set_checkpoint()

    a.vertical_flip()
    a.display()

    a.undo()
    a.display()

    print(a.checkpoints)
    a.revert_to_checkpoint('first')
    a.display()

def test_log_reconstruction():
    a = Glitcher()

    a.load_image(branches)
    a.rotate(1)
    a.invert_colors()
    a.save_image(dir + 'branches_logtest.png')
    a.rotate(3)
    b = Glitcher(log_file=(dir + 'branches_logtest.png.json'))
    b.display()
    a.display()

def test_noise():
    a = Glitcher()
    a.load_image(branches)
    a.jpeg_noise(2)
    a.display()

if __name__ == "__main__":
    test_noise()