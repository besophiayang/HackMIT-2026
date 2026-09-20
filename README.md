# HackMIT-2026

## Camera hand greeting

`hand_greeting.py` is a standalone controller that uses MediaPipe's dedicated
21-point hand model, shows an annotated live camera window, steers toward the
detected hand, and uses LiDAR to stop with braking margin near the person.
Existing emote files remain independent and unchanged.

Test perception without allowing movement first:

```bash
python hand_greeting.py 192.168.12.1 --observe-only
```

After observe-only detection is reliable, arm the slow approach with:

```bash
python hand_greeting.py 192.168.12.1 --approach-once
```

That single command performs at most one approach and then exits.

Only run one Go2 control program at a time. Keep the physical controller ready,
use a level uncluttered floor, and do not test near stairs, roads, pets, or crowds.
