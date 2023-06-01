
import sys 


def build_mol(self, dump_input=True, parse_arg=True,
              verbose=None, output=None, max_memory=None,
              atom=None, basis=None, unit=None, nucmod=None, ecp=None,
              charge=None, spin=0, symmetry=None, symmetry_subgroup=None,
              cart=None, magmom=None): # reduced time to generate from 150 ms to 1ms by removing gc.collect. could call it each 1000 molecules or so. 

        if sys.version_info >= (3,):
            unicode = str
            #print(unicode)
            #print("ASD")
            #exit()
        #exit()
        from pyscf import __config__ 
        DISABLE_GC = getattr(__config__, 'DISABLE_GC', False)

        if not DISABLE_GC and False:
            gc.collect()  # To release circular referred objects
            pass 

        if isinstance(dump_input, (str, unicode)):
            sys.stderr.write('Assigning the first argument %s to mol.atom\n' %
                             dump_input)
            dump_input, atom = True, dump_input

        if verbose is not None: self.verbose = verbose
        if output is not None: self.output = output
        if max_memory is not None: self.max_memory = max_memory
        if atom is not None: self.atom = atom
        if basis is not None: self.basis = basis
        if unit is not None: self.unit = unit
        if nucmod is not None: self.nucmod = nucmod
        if ecp is not None: self.ecp = ecp
        if charge is not None: self.charge = charge
        if spin != 0: self.spin = spin
        if symmetry is not None: self.symmetry = symmetry
        if symmetry_subgroup is not None: self.symmetry_subgroup = symmetry_subgroup
        if cart is not None: self.cart = cart
        if magmom is not None: self.magmom = magmom

        def _update_from_cmdargs_(mol):
            try:
                # Detect whether in Ipython shell
                __IPYTHON__  # noqa:
                return
            except Exception:
                pass

            if not mol._built: # parse cmdline args only once
                opts = cmd_args.cmd_args()

                if opts.verbose:
                    mol.verbose = opts.verbose
                if opts.max_memory:
                    mol.max_memory = opts.max_memory

                if opts.output:
                    mol.output = opts.output

        #if self.verbose >= logger.WARN:
        #    self.check_sanity()

        self._atom = self.format_atom(self.atom, unit=self.unit)
        uniq_atoms = set([a[0] for a in self._atom])

        if isinstance(self.basis, (str, unicode, tuple, list)):
            # specify global basis for whole molecule
            _basis = dict(((a, self.basis) for a in uniq_atoms))
        elif 'default' in self.basis:
            default_basis = self.basis['default']
            _basis = dict(((a, default_basis) for a in uniq_atoms))
            _basis.update(self.basis)
            del (_basis['default'])
        else:
            _basis = self.basis
        self._basis = self.format_basis(_basis)

        # TODO: Consider ECP info in point group symmetry initialization
        if self.ecp:
            # Unless explicitly input, ECP should not be assigned to ghost atoms
            if isinstance(self.ecp, (str, unicode)):
                _ecp = dict([(a, str(self.ecp))
                             for a in uniq_atoms if not is_ghost_atom(a)])
            elif 'default' in self.ecp:
                default_ecp = self.ecp['default']
                _ecp = dict(((a, default_ecp)
                             for a in uniq_atoms if not is_ghost_atom(a)))
                _ecp.update(self.ecp)
                del (_ecp['default'])
            else:
                _ecp = self.ecp
            self._ecp = self.format_ecp(_ecp)

        PTR_ENV_START   = 20
        env = self._env[:PTR_ENV_START]
        self._atm, self._bas, self._env = \
                self.make_env(self._atom, self._basis, env, self.nucmod,
                              self.nucprop)
        self._atm, self._ecpbas, self._env = \
                self.make_ecp_env(self._atm, self._ecp, self._env)

        if self.spin is None:
            self.spin = self.nelectron % 2
        else:
            # Access self.nelec in which the code checks whether the spin and
            # number of electrons are consistent.
            self.nelec

        if not self.magmom:
            self.magmom = [0.,]*self.natm
        import numpy 
        if self.spin == 0 and abs(numpy.sum(numpy.asarray(self.magmom)) - self.spin) > 1e-6:
            #don't check for unrestricted calcs.
            raise ValueError("mol.magmom is set incorrectly.")

        if self.symmetry:
            self._build_symmetry()

        #if dump_input and not self._built and self.verbose > logger.NOTE:
        #    self.dump_input()

        '''if self.verbose >= logger.DEBUG3:
            logger.debug3(self, 'arg.atm = %s', self._atm)
            logger.debug3(self, 'arg.bas = %s', self._bas)
            logger.debug3(self, 'arg.env = %s', self._env)
            logger.debug3(self, 'ecpbas  = %s', self._ecpbas)'''

        self._built = True
        return self