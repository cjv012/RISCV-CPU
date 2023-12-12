"""
RISCV cpu emulator

This is the sodor-1 stage CPU.

You will need to install pyelftoos to run this:
$ python3 -m pip install --user pyelftools

When complete, it will run:
- programs/fetch_test
- programs/simple
- programs/prog{N} for N in 1..5

Not supported are:
- branch instructions
- jump instructions
- stores to memory
- loads/stores
- CSR instructions
- syscalls / ecall

Author:
Date:
"""

import argparse
import itertools

# elf file loading
from elfloader import load_elf

# load the isa decoder
from isa import decode_instruction

# this gives us control and the enums for the control signals
from sodor_control import *

# load the gates used
from basicgates import gate
from mux import mux

from util import regNumToName
from twoscomp import int_to_tc

class Plus4(gate):
    "adds four to the input"
    def __init__(self, a):
        super().__init__()
        self.a = a

    def output(self):
        "output the input plus four"
        return self.a() + 4

class PC(gate):
    "on the clock, the PC is updated to the next PC"
    def __init__(self, nextpc, initialpc = 0):
        super().__init__()
        self.nextpc = nextpc
        self.currentpc = initialpc

    def output(self):
        "return the current pc value"
        return self.currentpc

    def clock(self):
        "set the current pc to the next pc"
        self.currentpc = self.nextpc()

class RegFile(gate):
    "RISCV register file"
    def __init__(self):
        super().__init__()
        # registers
        self.reg = [None] * 32
        # the zero register is always zero
        self.reg[0] = 0

    def __getitem__(self, idx):
        "return the value of the register at idx"
        return self.reg[idx]

    def __setitem__(self, idx, value):
        # don't allow writing to the zero register
		# this is complete, DO NOT CHANGE!
        if idx == 0:
            pass
        else:
            self.reg[idx] = value

    def clock(self, en, wa, wd):
        """
        Possibly write to a register, the arguments are: enable, write address, write data
        Note, do not allow writing to register zero.
        """
        if en and wa != 0:
            try:
                if wd is None:
                    print(f"REG WRITE reg[{regNumToName(wa)}] = None")
                else:
                    print(f"REG WRITE reg[{regNumToName(wa)}] = {wd:08x}")
                self.reg[wa] = wd
            except IndexError as x:
                raise IndexError(f"Register {self.wa()} is not a valid register.")

class RvCPU():
    "RISCV CPU emulator, memory is an ELFMemory instance (see load_elf)"
    def __init__(self, mem, pc = 0x80000000, quiet = False):
        # store the ELF memory in the cpu instance
        self.mem = mem

        # --------------------
        # construct the CPU architecture here (sodor 1 stage)
        # all CPU logic should appear here.
        # --------------------

        # use the pcmux to feed the correct next pc value into the pc register
        # since we haven't declared pcmux yet, we need to use a lambda to defer
        # the evaluation of the output of the pcmux until after the pcmux is
        # declared.  This is a common pattern in python.
        self.pc = PC(lambda: self.pcmux.output(), initialpc=pc)

        # a gate that adds four to the pc
        self.pc_plus4 = Plus4(self.pc.output)

        # the mux in front of the pc register selects the next pc value
        self.pcmux = mux(lambda: self.control['pc_sel'],
            self.pc_plus4.output,
            None, # jalr target
            None, # branch target
            None # jump target
            )

        # the op1 and op2 muxes select the operands for the ALU
        self.op1mux = mux(lambda: self.control['op1_sel'],
            lambda: self.reg[self.instruction.rs1],     # rs1
            lambda: self.instruction.z_imm,   # imz
            lambda: self.instruction.u_imm    # imu
            )

        self.op2mux = mux(lambda: self.control['op2_sel'],
            lambda: self.reg[self.instruction.rs2],     # rs2
            lambda: self.instruction.i_imm,             # imi
            lambda: self.instruction.s_imm,             # ims
            self.pc.output                              # pc
            )

        # the ALU is a mux of all the possible ALU operations
		# add is complete, the rest are TODO up to you!
        self.alu = mux(lambda: self.control['ALU_fun'],
            lambda: self.op1mux.output() + self.op2mux.output(), # add
            lambda: self.op1mux.output() ^ self.op2mux.output(), # xor
            lambda: self.op1mux.output(), # copy1
            lambda: self.op1mux.output() < self.op2mux.output(), # sltu
            lambda: self.op1mux.output() & self.op2mux.output(), # and
            lambda: self.op1mux.output() + self.op2mux.output(), # add
            lambda: self.op1mux.output() << self.op2mux.output(), # slt
            lambda: self.op1mux.output() >> self.op2mux.output(), # sra
            lambda: self.op1mux.output() - self.op2mux.output(), # sub
            lambda: self.op1mux.output() >> self.op2mux.output(), # srl
            lambda: self.op1mux.output() << self.op2mux.output(), # sll
            lambda: self.op1mux.output() | self.op2mux.output() # or
            )

        # the wb_sel mux selects the PC, result of the ALU or the memory
        # to be written back to the register file
        self.wb_selmux = mux(lambda: self.control['wb_sel'],
            self.pc_plus4.output,
            self.alu.output,
            lambda: self.mem[self.alu.output()]
            )

        # --------------------
        # end CPU architecture
        # --------------------

        # construct registers
        self.reg = RegFile()

        # instruction register (current executing instruction machine code value)
        self.ir = 0

        # misc options
        self.quiet = quiet


    def exec(self):
        """
        fetch from PC and execute the instruction.
        yields the cycle number of the instruction.
        """

        for cycle in itertools.count():
            # fetch the instruction from memory
            self.ir = self.mem[self.pc.output()]

            # decode the instruction
            self.instruction = decode_instruction(self.ir, self.pc.output(), symbols)

            # get the corresponding control signals for the instruction
            self.control = control[self.instruction.name]

            # check if we hit a null instruction (end of program)
            if self.instruction.val == 0:
                return
            else:
                # clock the regfile to write the result (if any) back to the register file
                self.reg.clock(self.control['rf_wen'], self.instruction.rd, self.wb_selmux.output())

                # yield the current instruction
                yield cycle

            # clock the CPU, this is the only clocked component in the one-stage CPU
            self.pc.clock()


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('elffiles', nargs="+",
        help="Binary instructions in ELF file format compiled for direct execution on the CPU.")
    parser.add_argument('--wordsize', type=int, default=4)
    parser.add_argument("-q", "--quiet",
        help="Quieter output (no extra info) [default=True]",
        action="store_true", default=False)

    args = parser.parse_args()

    # allow multiple files from the CLI
    for elffile in args.elffiles:

        print('='*30 + f'<{elffile.center(20)}>' + '='*30)

        try:
            # load the elf file and symbol table into memory
            sys_mem, symbols = load_elf(elffile, quiet = False)
        except Exception as x:
            print(f"ERR - couldn't load {elffile}.")
            raise(x)

        print("-"*60)
        cpu = RvCPU(sys_mem, pc=symbols['_start'], quiet = args.quiet)
        fmtreg = lambda x: f"{int_to_tc(x):08x}" if x != None else "*"*8
        for cycle in cpu.exec():
            if not args.quiet:
                print (f"{cycle:8d}: PC={fmtreg(cpu.pc.output())}, IR={cpu.ir:08x}, {cpu.instruction}")
        print("-"*60)

        # print the registers

        print("Final register values:")
        for i in range(0,32,8):
            print(" ".join([f"{regNumToName(i+j).rjust(3)}: {fmtreg(cpu.reg[i+j])}" for j in range(8)]))


