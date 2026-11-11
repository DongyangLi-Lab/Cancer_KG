"""Process-local policy for existing Flash2.8.3 varlen function only."""
import ast,functools,hashlib,importlib,importlib.metadata,inspect,textwrap
from pathlib import Path
class FlashCallPolicy:
    def __init__(self,module,version,force_deterministic):
        if version!='2.8.3':raise ValueError('This bounded wrapper requires verified flash-attn2.8.3')
        self.module=module;self.version=version;self.force=bool(force_deterministic);self.original=module.flash_attn_varlen_func;self.signature=inspect.signature(self.original)
        parameter=self.signature.parameters.get('deterministic')
        if parameter is None or parameter.default is not False or parameter.kind not in [inspect.Parameter.POSITIONAL_OR_KEYWORD,inspect.Parameter.KEYWORD_ONLY]:raise ValueError('Verified deterministic=False function signature required')
        if not {'q','k','v','cu_seqlens_q','cu_seqlens_k','max_seqlen_q','max_seqlen_k'}<=set(self.signature.parameters):raise ValueError('Native varlen interface changed')
        self.calls=0;self.effective_true=0;self.effective_false=0;self.forced_overrides=0;self.omitted=0;self.restored=False;self.entered=False
        path=inspect.getsourcefile(self.original);self.source_file=path;self.source_hash=hashlib.sha256(Path(path).read_bytes()).hexdigest() if path and Path(path).is_file() else None
        @functools.wraps(self.original)
        def wrapper(*args,**kwargs):
            bound=self.signature.bind(*args,**kwargs);before={k:v for k,v in bound.arguments.items() if k!='deterministic'}
            value=bound.arguments.get('deterministic',False);self.omitted+=int('deterministic' not in bound.arguments)
            if self.force:self.forced_overrides+=int(value is not True);bound.arguments['deterministic']=True
            if not all(bound.arguments[k] is value for k,value in before.items()):raise RuntimeError('Wrapper altered a nonpolicy parameter')
            effective=bound.arguments.get('deterministic',False);self.calls+=1;self.effective_true+=int(effective is True);self.effective_false+=int(effective is False)
            return self.original(*bound.args,**bound.kwargs)
        self.wrapper=wrapper
    def __enter__(self):
        if self.entered or self.module.flash_attn_varlen_func is not self.original:raise RuntimeError('Function changed or wrapper already entered')
        self.module.flash_attn_varlen_func=self.wrapper;self.entered=True;self.restored=False;return self
    def __exit__(self,kind,value,trace):
        if self.module.flash_attn_varlen_func is not self.wrapper:raise RuntimeError('Unexpected function replacement during bounded policy')
        self.module.flash_attn_varlen_func=self.original;self.restored=True;self.entered=False;return False
    def summary(self):
        unchanged=self.source_hash is None or hashlib.sha256(Path(self.source_file).read_bytes()).hexdigest()==self.source_hash
        return dict(version=self.version,mode='force existing deterministic=True only' if self.force else 'observe original native deterministic=False default',signature=str(self.signature),calls=self.calls,effective_deterministic_true_calls=self.effective_true,effective_deterministic_false_calls=self.effective_false,omitted_argument_calls=self.omitted,deterministic_overrides=self.forced_overrides,other_arguments_unchanged=True,source_file_sha256=self.source_hash,source_file_unchanged=unchanged,original_function_restored=self.restored)
def native_flash_policy(force_deterministic):
    interface=importlib.import_module('flash_attn.flash_attn_interface');package=importlib.import_module('flash_attn');attention=importlib.import_module('llmfoundry.models.layers.attention')
    # This installed llmfoundry imports the interface locally inside flash_attn_fn.
    if package.flash_attn_interface is not interface:raise RuntimeError('Flash package does not reference the verified interface module')
    tree=ast.parse(textwrap.dedent(inspect.getsource(attention.flash_attn_fn)))
    local_import=any(isinstance(n,ast.ImportFrom) and n.module=='flash_attn' and any(a.name=='flash_attn_interface' and a.asname is None for a in n.names) for n in ast.walk(tree))
    native_call=any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='flash_attn_interface' and n.func.attr=='flash_attn_varlen_func' for n in ast.walk(tree))
    if not local_import or not native_call:raise RuntimeError('Native llmfoundry Flash local import/call route changed')
    policy=FlashCallPolicy(interface,importlib.metadata.version('flash-attn'),force_deterministic)
    if policy.source_hash is None:raise RuntimeError('Cannot establish original Flash interface source provenance')
    return policy
