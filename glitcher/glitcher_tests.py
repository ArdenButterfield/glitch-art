from glitcher import Glitcher

dir = '../images/raw/'
branches = dir + 'branches.jpg'
t1_png = dir + 'test1.png'
flag = dir + 'pride_flag.png'

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

def test_numpy_shuffle():
    a = Glitcher()
    a.load_image(branches)
    a.shuffle(0, chunks=18, random_order=False, even_slices=True)
    a.display()

def test_jpeg_shuffle():
    a = Glitcher()
    a.load_image(branches)

    a.shuffle(2,chunks=20, even_slices=True, entire_image=False)
    a.display()

def test_bitflip():
    a = Glitcher()
    a.load_image(branches)
    a.jpeg_bit_flip(20, True)
    # a.jpeg_noise(100)
    a.display()

def test_flip_all_bits():
    a = Glitcher()
    a.load_image(branches)
    a.flip_all_the_bits()
    a.display()

def test_waves():
    a = Glitcher()
    a.load_image(branches)
    dims = a.save_wav('test.wav', 0)
    a.read_wav('test.wav', 0, dims)
    a.display()
    dims = a.save_wav('test.wav', 1)
    a.read_wav('test.wav', 1, dims)
    a.display()
    dims = a.save_wav(['t0.wav', 't1.wav', 't2.wav'], 2)
    a.read_wav(['t0.wav', 't1.wav', 't2.wav'], 2, dims)
    a.display()

def test_waves_fun():
    a = Glitcher()
    a.load_image(branches)
    for i in range(4):
        dims = a.save_wav('test.wav', 1)
        a.read_wav('test.wav', 0, dims)
    a.display()

def test_bug_eater():

    a = Glitcher()
    a.load_image(t1_png)
    b = a.to_bug_eater(0)

def test_automata():
    a = Glitcher()
    a.load_image(dir + 'seagull.jpg')
    a.elementary_automata(154)
    a.save_image(dir+'automata_seagull.png')

def test_bayer():
    a = Glitcher()
    a.load_image('/Volumes/Lasagna/SCHOOL/Images/observe classmate/gamer.png')
    a.bayer_filter(2)
    a.save_image('/Volumes/Lasagna/SCHOOL/Images/observe classmate/gamerbayercolor_noinvert.png')

def test_grayscale():
    a = Glitcher()
    a.load_image(branches)
    a.make_grayscale()
    a.display()

def test_load_binary():
    a = Glitcher()
    a.load_binary(dir + 'bfs',128,128,grayscale=False)
    a.display()

def test_flatten_reshape():
    for i in range(3):
        a = Glitcher()
        a.load_image(t1_png)
        a.flatten_reshape((1000,1000),i)
        a.display()

def test_fly_eye():
    a = Glitcher()
    a.load_image(branches)
    a.fly_eye((150,100),0.1,y_backwards=False, x_backwards=False, in_place=False)
    a.display()

def test_extreme():
    a = Glitcher()
    a.load_image(branches)
    a.pixel_extreme(-1)
    a.display()

def test_edge_detect():
    a = Glitcher()
    a.load_image(branches)
    a.edge_detect()
    a.display()
    a.load_image(branches)
    a.display()

def test_2d_auto():
    a = Glitcher()
    a.load_image(branches)
    a.display()
    a.cellular_2d_automata(['*****'])
    a.display()


if __name__ == "__main__":
    test_2d_auto()