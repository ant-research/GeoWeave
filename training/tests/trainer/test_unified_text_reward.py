import json
from types import SimpleNamespace

import torch

from unirl.models.sensenova_u1.conditions import SensenovaU1ARConditions
from unirl.trainer.unified_model import UnifiedModelTrainer
from unirl.types.primitives import Texts
from unirl.types.prompts import RolloutInputs
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack, _track_with_field
from unirl.types.sampling import ARSamplingParams
from unirl.types.segments import TextSegment


class _Reward:
    def __init__(self):
        self.req = None
        self.track = None

    def score_and_attach(self, *, req, track):
        self.req = req
        self.track = track
        return _track_with_field(track, "rewards", torch.tensor([1.0, 0.0, 0.0, 1.0]))


class _Stack:
    def __init__(self):
        self.args = None

    def train_track(
        self, ar_track, image_track, *, training_progress, force_optimizer_step=False
    ):
        self.args = (ar_track, image_track, training_progress)
        return {
            "ar": SimpleNamespace(
                loss=0.0, metrics={}, has_backward=True, grad_norm=0.0, lr=0.0
            )
        }


class _Logger:
    def should_log_media(self, _rollout_id):
        return False

    def log_rollout_step(self, *_args, **_kwargs):
        pass


def test_text_reward_scores_ar_output_and_aligns_metadata():
    segment = TextSegment.pack(
        tokens=[torch.tensor([1]), torch.tensor([2]), torch.tensor([3]), torch.tensor([4])],
        log_probs=[torch.zeros(1) for _ in range(4)],
    )
    ar_track = RolloutTrack(
        sample_ids=["p0/a0", "p0/a1", "p1/a0", "p1/a1"],
        parent_ids=["p0", "p0", "p1", "p1"],
        segment=segment,
        decoded=Texts(texts=["answer 1", "answer 2", "answer 3", "answer 4"]),
    )
    req = RolloutReq(
        sample_ids=["p0", "p1"],
        group_ids=["g0", "g1"],
        primitives={"text": Texts(texts=["question 0", "question 1"])},
        sampling_params={"ar": ARSamplingParams(samples_per_prompt=2, max_new_tokens=8)},
        metadata=[{"answer": "1"}, {"answer": "4"}],
    )
    trainer = UnifiedModelTrainer.__new__(UnifiedModelTrainer)
    trainer.reward_track = "ar"
    trainer._single_engine = True
    trainer._rollout_reward_overlap_enabled = False
    trainer._shared_advantage = True
    trainer._enable_fsdp_offload = False
    trainer.dump_dir = None
    trainer.rollout = SimpleNamespace(generate=lambda _req: RolloutResp(tracks={"ar": ar_track}))
    trainer.reward = _Reward()
    trainer.stack = _Stack()
    trainer.wandb_logger = _Logger()

    results, mean_reward = trainer.train_step(req)

    assert list(results) == ["ar"]
    assert mean_reward == 0.5
    assert trainer.reward.track.decoded.texts == ["answer 1", "answer 2", "answer 3", "answer 4"]
    assert trainer.reward.req.primitives["text"].texts == ["question 0", "question 0", "question 1", "question 1"]
    assert trainer.reward.req.metadata == [{"answer": "1"}, {"answer": "1"}, {"answer": "4"}, {"answer": "4"}]
    trained_ar, trained_image, _ = trainer.stack.args
    assert trained_ar.advantages is not None
    assert trained_image is None


def test_build_req_supports_ar_only_sampling_and_forwards_stage_config():
    trainer = UnifiedModelTrainer.__new__(UnifiedModelTrainer)
    trainer.sampling_params = {"ar": ARSamplingParams(samples_per_prompt=2, max_new_tokens=8)}
    trainer._stage_config = {"system_message": "geometry"}
    inputs = RolloutInputs(
        sample_ids=["p0"],
        group_ids=["g0"],
        primitives={"text": Texts(texts=["question"])},
        metadata=[{"answer": "2"}],
    )

    req = trainer._build_req(inputs, rollout_id=0)

    assert list(req.sampling_params) == ["ar"]
    assert req.stage_config == {"system_message": "geometry"}


def test_ar_rollout_dump_records_text_length_reward_and_truncation(tmp_path):
    segment = TextSegment.pack(
        tokens=[torch.tensor([1, 2]), torch.arange(8)],
        log_probs=[torch.zeros(2), torch.zeros(8)],
    )
    ar_track = RolloutTrack(
        sample_ids=["p0/a0", "p0/a1"],
        parent_ids=["p0", "p0"],
        segment=segment,
        decoded=Texts(texts=["short answer", "unfinished reasoning"]),
        rewards=torch.tensor([1.0, 0.0]),
        advantages=torch.tensor([1.0, -1.0]),
    )
    req = RolloutReq(
        sample_ids=["p0"],
        group_ids=["geometry-0"],
        primitives={"text": Texts(texts=["question 0"])},
        sampling_params={"ar": ARSamplingParams(samples_per_prompt=2, max_new_tokens=8)},
        metadata=[{"answer": "42"}],
    )
    trainer = UnifiedModelTrainer.__new__(UnifiedModelTrainer)
    trainer.dump_dir = str(tmp_path)
    trainer.reward_track = "ar"
    trainer.sampling_params = req.sampling_params

    trainer._dump_rollout(3, req, RolloutResp(tracks={"ar": ar_track}))

    dump_path = tmp_path / "rollout_3" / "samples.jsonl"
    records = [json.loads(line) for line in dump_path.read_text().splitlines()]
    assert records[0]["generated_text"] == "short answer"
    assert records[0]["response_length"] == 2
    assert records[0]["gold_answer"] == "42"
    assert records[0]["reward"] == 1.0
    assert records[0]["advantage"] == 1.0
    assert records[0]["truncated"] is False
    assert records[1]["response_length"] == 8
    assert records[1]["truncated"] is True


def test_ar_rollout_dump_saves_ragged_auxiliary_images_and_boundaries(tmp_path):
    segment = TextSegment.pack(
        tokens=[torch.tensor([1, 9, 2]), torch.tensor([3])],
        log_probs=[torch.zeros(3), torch.zeros(1)],
    )
    conditions = SensenovaU1ARConditions(
        prompt_queries=["q0", "q1"],
        pixel_values=[None, None],
        grid_hws=[None, None],
        generated_images=[
            [torch.zeros(3, 32, 64), torch.ones(3, 64, 32)],
            [],
        ],
        text_segment_boundaries=[[(0, 2), (2, 3)], [(0, 1)]],
    )
    ar_track = RolloutTrack(
        sample_ids=["p0/a0", "p0/a1"],
        parent_ids=["p0", "p0"],
        segment=segment,
        decoded=Texts(texts=["with images", "without image"]),
        conditions=conditions.to_dict(),
        rewards=torch.tensor([0.0, 1.0]),
        advantages=torch.tensor([-1.0, 1.0]),
    )
    req = RolloutReq(
        sample_ids=["p0"],
        group_ids=["geometry-0"],
        primitives={"text": Texts(texts=["question 0"])},
        sampling_params={"ar": ARSamplingParams(samples_per_prompt=2, max_new_tokens=8)},
        metadata=[{"answer": "42"}],
    )
    trainer = UnifiedModelTrainer.__new__(UnifiedModelTrainer)
    trainer.dump_dir = str(tmp_path)
    trainer.reward_track = "ar"
    trainer.sampling_params = req.sampling_params

    trainer._dump_rollout(4, req, RolloutResp(tracks={"ar": ar_track}))

    out_dir = tmp_path / "rollout_4"
    records = [json.loads(line) for line in (out_dir / "samples.jsonl").read_text().splitlines()]
    assert records[0]["generated_image_count"] == 2
    assert records[0]["generated_image_shapes"] == [[3, 32, 64], [3, 64, 32]]
    assert records[0]["text_segment_boundaries"] == [[0, 2], [2, 3]]
    assert records[1]["generated_image_count"] == 0
    for image_file in records[0]["generated_image_files"]:
        assert (out_dir / image_file).is_file()


def test_ar_rollout_dump_can_skip_pngs_but_keep_jsonl(tmp_path):
    segment = TextSegment.pack(tokens=[torch.tensor([1])], log_probs=[torch.zeros(1)])
    conditions = SensenovaU1ARConditions(
        prompt_queries=["q0"],
        pixel_values=[None],
        grid_hws=[None],
        generated_images=[[torch.zeros(3, 32, 64)]],
        text_segment_boundaries=[[(0, 1)]],
    )
    ar_track = RolloutTrack(
        sample_ids=["p0/a0"],
        parent_ids=["p0"],
        segment=segment,
        decoded=Texts(texts=["answer"]),
        conditions=conditions.to_dict(),
        rewards=torch.tensor([0.0]),
        advantages=torch.tensor([0.0]),
        component_rewards={"judge_failed": torch.tensor([1.0])},
    )
    req = RolloutReq(
        sample_ids=["p0"],
        group_ids=["geometry-0"],
        primitives={"text": Texts(texts=["question"])},
        sampling_params={"ar": ARSamplingParams(samples_per_prompt=1, max_new_tokens=8)},
        metadata=[{"answer": "42"}],
    )
    trainer = UnifiedModelTrainer.__new__(UnifiedModelTrainer)
    trainer.dump_dir = str(tmp_path)
    trainer.reward_track = "ar"
    trainer.sampling_params = req.sampling_params
    trainer.save_rollout_results = True
    trainer.save_rollout_images = False

    trainer._dump_rollout(5, req, RolloutResp(tracks={"ar": ar_track}))

    out_dir = tmp_path / "rollout_5"
    record = json.loads((out_dir / "samples.jsonl").read_text().strip())
    assert record["generated_image_count"] == 1
    assert record["generated_image_files"] == []
    assert record["judge_failed"] == 1.0
    assert list(out_dir.glob("*.png")) == []


def test_ar_rollout_dump_can_skip_jsonl_but_keep_pngs(tmp_path):
    segment = TextSegment.pack(tokens=[torch.tensor([1])], log_probs=[torch.zeros(1)])
    conditions = SensenovaU1ARConditions(
        prompt_queries=["q0"],
        pixel_values=[None],
        grid_hws=[None],
        generated_images=[[torch.zeros(3, 32, 32)]],
        text_segment_boundaries=[[(0, 1)]],
    )
    track = RolloutTrack(
        sample_ids=["p0/a0"],
        parent_ids=["p0"],
        segment=segment,
        decoded=Texts(texts=["answer"]),
        conditions=conditions.to_dict(),
        rewards=torch.tensor([1.0]),
        advantages=torch.tensor([0.0]),
    )
    req = RolloutReq(
        sample_ids=["p0"],
        group_ids=["g0"],
        primitives={"text": Texts(texts=["question"])},
        sampling_params={"ar": ARSamplingParams(samples_per_prompt=1, max_new_tokens=8)},
        metadata=[{"answer": "42"}],
    )
    trainer = UnifiedModelTrainer.__new__(UnifiedModelTrainer)
    trainer.dump_dir = str(tmp_path)
    trainer.reward_track = "ar"
    trainer.sampling_params = req.sampling_params
    trainer.save_rollout_results = False
    trainer.save_rollout_images = True

    trainer._dump_rollout(6, req, RolloutResp(tracks={"ar": track}))

    out_dir = tmp_path / "rollout_6"
    assert not (out_dir / "samples.jsonl").exists()
    assert (out_dir / "sample_0_aux_0.png").is_file()
