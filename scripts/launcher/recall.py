"""Typed recall options shared by launch plans and model-free worker checks."""

LAYERWISE_RESOURCES = {
    'LayerwiseActivationRamMiB': ('--kvmem-layerwise-activation-ram-mib', 256),
    'LayerwiseActivationDiskMiB': ('--kvmem-layerwise-activation-disk-mib', 0),
    'LayerwiseStagingMiB': ('--kvmem-layerwise-staging-mib', 16),
    'LayerwiseMetadataMiB': ('--kvmem-layerwise-metadata-mib', 256),
    'LayerwiseActivationMetadataMiB': ('--kvmem-layerwise-activation-metadata-mib', 128),
}
MTP_HISTORY_POLICIES = ('upstream-global', 'first-layer')


def mtp_history_default(spec):
    """Only the selected worker may declare a default for its own MTP path."""
    if 'default_history_policy' not in spec:
        return None
    value = spec['default_history_policy']
    policies = spec.get('history_policies')
    if value not in MTP_HISTORY_POLICIES or not isinstance(policies, list) or value not in policies:
        raise ValueError('Invalid worker MTP history default')
    return value


def resource_defaults(profile):
    """Read optional backend-owned byte limits; absence is not a guessed value."""
    if 'default_resources' not in profile:
        return {}
    supplied = profile['default_resources']
    if not isinstance(supplied, dict):
        raise ValueError('Invalid worker resource defaults')
    values = {}
    for key, (flag, _) in LAYERWISE_RESOURCES.items():
        native = flag.removeprefix('--kvmem-layerwise-').replace('-', '_').replace('_mib', '_bytes')
        if native not in supplied:
            continue
        value = supplied[native]
        if (type(value) is not int or not 0 <= value <= (2147483647 << 20)
                or (key not in ('LayerwiseActivationRamMiB', 'LayerwiseActivationDiskMiB') and value == 0)):
            raise ValueError('Invalid worker resource defaults: ' + native)
        values[key] = value
    return values


def token_constraints(profile, spec=None, concurrency=1):
    """Optional worker limits; omitted fields retain the older boolean contract."""
    def limit(owner, key):
        if key not in owner:
            return None
        value = owner[key]
        if type(value) is not int or not 1 <= value <= 2147483647:
            raise ValueError('Invalid worker token constraint: ' + key)
        return value
    minimum = limit(profile, 'minimum_window_tokens')
    spec = {} if spec is None else spec
    maximum = limit(spec, 'max_ngram_draft_tokens')
    concurrent = limit(spec, 'max_ngram_draft_tokens_concurrent')
    if concurrency > 1 and concurrent is not None:
        maximum = concurrent if maximum is None else min(maximum, concurrent)
    return minimum, maximum


def require_ngram_limit(profile, spec, parameters):
    _, maximum = token_constraints(profile, spec, parameters.get('Concurrency', 1))
    requested = parameters.get('NgramDrafts', 0)
    if maximum is not None and requested > maximum:
        raise ValueError(f'This worker limits layerwise NgramDrafts to {maximum} at the selected concurrency; requested {requested}')


def strategy(parameters):
    return parameters.get('RecallStrategy', 'legacy')


def speculative_mode(parameters):
    return parameters.get('SpeculativeMode') or ('mtp' if parameters.get('MtpDrafts', 0) else 'none')


def validate_recall(parameters):
    selected = strategy(parameters)
    if selected not in ('legacy', 'layerwise'):
        raise ValueError('RecallStrategy must be legacy or layerwise')
    if 'LayerwiseCheckpointTarget' in parameters:
        target = parameters['LayerwiseCheckpointTarget']
        if selected != 'layerwise' or type(target) is not int or not 1 <= target <= 2147483647:
            raise ValueError('LayerwiseCheckpointTarget requires layerwise recall and a positive integer')
    history = parameters.get('LayerwiseMtpHistory')
    if speculative_mode(parameters) == 'guess' and (selected != 'layerwise' or parameters.get('Backend') != 'polystrata'):
        raise ValueError('Guess is the layerwise PolyStrata Gemma shared-target draft mode')
    if history is not None and history not in MTP_HISTORY_POLICIES:
        raise ValueError('LayerwiseMtpHistory must be upstream-global or first-layer')
    if history is not None and (selected != 'layerwise' or speculative_mode(parameters) != 'mtp'):
        raise ValueError('LayerwiseMtpHistory requires layerwise recall and native MTP')
    if selected != 'layerwise':
        if set(LAYERWISE_RESOURCES) & parameters.keys():
            raise ValueError('Layerwise resource limits require RecallStrategy=layerwise')
        return
    values = {key: parameters[key] for key in LAYERWISE_RESOURCES if key in parameters}
    for key, value in values.items():
        if type(value) is not int or not 0 <= value <= 2147483647:
            raise ValueError(f'{key} must be a finite nonnegative MiB integer')
    if values.get('LayerwiseActivationRamMiB') == 0 and values.get('LayerwiseActivationDiskMiB') == 0:
        raise ValueError('Layerwise activation RAM or SSD budget must be positive')
    for key in ('LayerwiseStagingMiB', 'LayerwiseMetadataMiB', 'LayerwiseActivationMetadataMiB'):
        if key in values and not values[key]:
            raise ValueError(f'{key} must be positive')


def recall_args(parameters):
    validate_recall(parameters)
    if strategy(parameters) == 'legacy':
        return []
    args = ['--kvmem-recall-strategy', 'layerwise']
    for key, (flag, _) in LAYERWISE_RESOURCES.items():
        if key in parameters:
            args.extend((flag, str(parameters[key])))
    if 'LayerwiseCheckpointTarget' in parameters:
        args.extend(('--kvmem-layerwise-checkpoint-target', str(parameters['LayerwiseCheckpointTarget'])))
    if 'LayerwiseMtpHistory' in parameters:
        args.extend(('--kvmem-layerwise-mtp-history', parameters['LayerwiseMtpHistory']))
    return args


def require_recall_features(capabilities, parameters):
    """Source support or a legacy revision never qualifies a worker's new mode."""
    validate_recall(parameters)
    if strategy(parameters) != 'layerwise':
        return
    revision = capabilities.get('recall_strategy_revision')
    profiles = capabilities.get('recall_strategies', {})
    profile = profiles.get('layerwise', {}) if isinstance(profiles, dict) else {}
    if (type(revision) is not int or revision < 1 or not isinstance(profile, dict)
            or profile.get('available') is not True or profile.get('current_layer_q') is not True
            or profile.get('query_replay') is not False or profile.get('position_policy') != 'original'):
        raise ValueError('This worker has no qualified layerwise recall contract; update the complete backend package')
    if 'LayerwiseCheckpointTarget' in parameters:
        retention = profile.get('checkpoint_retention')
        maximum = retention.get('max_target') if isinstance(retention, dict) else None
        if type(maximum) is not int or not 1 <= parameters['LayerwiseCheckpointTarget'] <= maximum:
            raise ValueError('The selected worker does not support the requested checkpoint target')
    defaults = resource_defaults(profile)
    minimum_window, _ = token_constraints(profile)
    if minimum_window is not None:
        budget, reserve = parameters.get('Budget'), parameters.get('Reserve')
        if type(budget) is not int or type(reserve) is not int:
            raise ValueError('Worker minimum_window_tokens requires resolved Budget and Reserve')
        if budget + reserve < minimum_window:
            raise ValueError(f'This worker requires Budget + Reserve >= {minimum_window} tokens; requested {budget + reserve}')
    effective_storage = {key: parameters[key] << 20 if key in parameters else defaults.get(key)
                         for key in ('LayerwiseActivationRamMiB', 'LayerwiseActivationDiskMiB')}
    if all(value == 0 for value in effective_storage.values()):
        raise ValueError('Layerwise activation RAM or SSD budget must be positive')
    architectures = profile.get('model_architectures')
    if architectures is not None:
        if (not isinstance(architectures, list) or not architectures
                or any(not isinstance(item, str) or not item for item in architectures)):
            raise ValueError('This worker has no qualified layerwise model architectures')
        from model_health import Inspection
        from pathlib import Path
        inspected = Inspection(parameters, Path.cwd()).run()
        model = inspected.get('model', {})
        architecture = model.get('architecture_id', model.get('architecture')
                                 if model.get('format') == 'GGUF' else None)
        if not inspected.get('can_start') or architecture not in architectures:
            raise ValueError('This worker does not qualify the selected layerwise model architecture '
                             + str(architecture or '(unverified)') + '; qualified architectures: '
                             + ', '.join(architectures))
    for name, value in (('max_devices', len(parameters.get('Gpu', '').split(','))),
                        ('max_concurrency', parameters.get('Concurrency', 1))):
        limit = profile.get(name)
        if type(limit) is not int or limit < 1 or value > limit:
            raise ValueError(f'This worker does not qualify layerwise {name}')
    floor = profile.get('minimum_reserve_blocks')
    block = 64 if parameters.get('Backend') == 'ninfer' else 128 if parameters.get('Backend') == 'radiance' else parameters.get('BlockTokens', 32)
    reserve = parameters.get('Reserve', 1024)
    if type(floor) is not int or floor < 1 or type(block) is not int or block < 1:
        raise ValueError('This worker requires an explicit qualified layerwise reserve floor')
    if reserve < floor * block:
        raise ValueError(f'This worker has a layerwise reserve floor of {floor * block} tokens ({floor} blocks); requested {reserve}')
    devices = len(parameters.get('Gpu', '').split(','))
    kind = 'single' if devices == 1 else 'tensor' if parameters.get('Backend') == 'radiance' or parameters.get('TensorParallel') else 'pipeline'
    parallelism = profile.get('parallel_modes', [])
    if not isinstance(parallelism, list) or kind not in parallelism:
        raise ValueError(f'This worker does not qualify layerwise parallelism {kind}; qualified modes: {parallelism}')
    formats = profile.get('kv_formats')
    if not isinstance(formats, list) or not formats or any(not isinstance(item, str) for item in formats):
        raise ValueError('This worker has no qualified layerwise KV formats')
    aliases = {'fp16': 'f16', 'fp32': 'f32', 'q8': 'q8_0', 'q5': 'q5_0', 'q4': 'q4_0'}
    for key in ('KvTypeK', 'KvTypeV') if parameters.get('Backend') == 'llamacpp' else ('KvType',):
        value = parameters.get(key) or parameters.get('KvType') or ('q8_0' if parameters.get('Backend') == 'llamacpp' else 'int8')
        if aliases.get(value, value) not in formats:
            raise ValueError(f'This worker does not qualify layerwise {key}={value}; qualified formats: {", ".join(formats)}')
    vision = profile.get('vision_residency', [])
    if not isinstance(vision, list) or any(not isinstance(item, str) for item in vision):
        raise ValueError('This worker has an invalid layerwise vision contract')
    if parameters.get('Vision') and parameters.get('VisionResidency', 'resident') not in vision:
        raise ValueError('This worker does not qualify the selected layerwise vision residency')
    mode = speculative_mode(parameters)
    modes = profile.get('speculative', {})
    spec = modes.get(mode, {}) if isinstance(modes, dict) else {}
    if not isinstance(spec, dict) or spec.get('available') is not True:
        raise ValueError(f'This worker does not qualify layerwise speculative mode {mode}')
    if mode == 'none':
        return
    require_ngram_limit(profile, spec, parameters)
    count = parameters.get('DraftTokens', parameters.get('MtpDrafts', 0))
    maximum = spec.get('max_draft_tokens')
    if type(count) is not int or type(maximum) is not int or maximum < 1 or not 1 <= count <= maximum:
        raise ValueError(f'This worker does not qualify layerwise draft width {count}; qualified maximum: {maximum}')
    if (parameters.get('DraftPolicy') == 'adaptive' or parameters.get('AdaptiveMtp')) and spec.get('adaptive') is not True:
        raise ValueError('This worker does not qualify layerwise adaptive drafts')
    if parameters.get('NgramDrafts', 0) and spec.get('ngram') is not True:
        raise ValueError('This worker does not qualify layerwise N-gram drafts')
    if mode == 'mtp':
        policies = spec.get('history_policies', [])
        selected_history = parameters.get('LayerwiseMtpHistory')
        default_history = mtp_history_default(spec)
        if selected_history is None:
            selected_history = default_history
        if selected_history is None:
            raise ValueError('This worker requires an explicit history policy; no default has been declared')
        if not isinstance(policies, list) or selected_history not in policies:
            raise ValueError('This worker does not qualify the selected layerwise MTP history policy')
        state = 'replay' if parameters.get('Backend') == 'llamacpp' else 'native'
        states = spec.get('state_modes', [])
        if not isinstance(states, list) or state not in states:
            raise ValueError('This worker does not qualify the layerwise MTP native state mode')
        per_policy = spec.get('history_policy_caps', {})
        if not isinstance(per_policy, dict):
            raise ValueError('This worker has invalid MTP policy capabilities')
        if selected_history in per_policy:
            narrowed = per_policy[selected_history]
            if not isinstance(narrowed, dict):
                raise ValueError('This worker has invalid MTP policy capabilities')
            policy_spec = dict(spec, **narrowed)
            require_ngram_limit(profile, policy_spec, parameters)
            if policy_spec.get('available') is not True:
                raise ValueError('This worker does not qualify the selected MTP history policy')
            if type(policy_spec.get('max_draft_tokens')) is not int or count > policy_spec['max_draft_tokens']:
                raise ValueError('This worker does not qualify this MTP policy draft width')
            if (parameters.get('DraftPolicy') == 'adaptive' or parameters.get('AdaptiveMtp')) and policy_spec.get('adaptive') is not True:
                raise ValueError('This worker does not qualify this MTP policy adaptive drafts')
            policy_states = policy_spec.get('state_modes', [])
            if not isinstance(policy_states, list) or state not in policy_states:
                raise ValueError('This worker does not qualify this MTP policy native state mode')
            for key, value in (('max_devices', devices), ('max_concurrency', parameters.get('Concurrency', 1))):
                limit = narrowed.get(key, profile[key])
                if type(limit) is not int or limit < 1 or value > limit:
                    raise ValueError(f'This worker does not qualify this MTP policy {key}')
            if parameters.get('NgramDrafts', 0) and policy_spec.get('ngram') is not True:
                raise ValueError('This worker does not qualify this MTP policy N-gram drafts')
    if mode == 'dflash2' and spec.get('history') != 'independent-native-window':
        raise ValueError('This worker does not qualify the independent layerwise DFlash2 window')
    if mode == 'guess':
        if spec.get('history') != 'shared-target' or spec.get('model_architectures') != ['gemma4']:
            raise ValueError('This worker does not qualify the Gemma shared-target Guess path')
        from model_health import Inspection
        from pathlib import Path
        inspected = Inspection(parameters, Path.cwd()).run()
        if inspected.get('model', {}).get('architecture') != 'gemma4' or not inspected.get('can_start'):
            raise ValueError('Guess requires a trusted, compatible Gemma GGUF model')
    if mode == 'mtp' and parameters.get('Backend') == 'polystrata':
        from model_health import Inspection
        from pathlib import Path
        inspected = Inspection(parameters, Path.cwd()).run()
        if inspected.get('model', {}).get('architecture') != 'qwen35moe' or not inspected.get('can_start'):
            raise ValueError('PolyStrata OWN MTP requires a trusted Qwen GGUF; Gemma uses shared-target Guess')
