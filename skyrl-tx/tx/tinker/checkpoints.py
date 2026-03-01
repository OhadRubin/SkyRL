"""Checkpoint management utilities for the Tinker API."""

import shutil

from fastapi import HTTPException, Request
from sqlmodel import select, Session
from sqlmodel.ext.asyncio.session import AsyncSession

from tx.tinker import types
from tx.tinker.config import EngineConfig
from tx.tinker.db_models import CheckpointDB, CheckpointStatus, ModelDB
from observability import log

# Default for max sampler checkpoints (actual value comes from EngineConfig)
_DEFAULT_MAX_SAMPLER_CHECKPOINTS = 3
LATEST_CHECKPOINT_PREFIX = "latest_"
MAX_LATEST_TRAINING_CHECKPOINTS = 1


async def create_checkpoint(
    session: AsyncSession,
    model_id: str,
    checkpoint_id: str,
    checkpoint_type: types.CheckpointType,
):
    """Create a pending CheckpointDB entry, relying on database constraints for validation."""
    from sqlalchemy.exc import IntegrityError

    checkpoint_db = CheckpointDB(
        model_id=model_id,
        checkpoint_id=checkpoint_id,
        checkpoint_type=checkpoint_type,
        status=CheckpointStatus.PENDING,
    )
    session.add(checkpoint_db)

    try:
        await session.flush()
    except IntegrityError:
        await session.rollback()
        # Check if the model exists
        statement = select(ModelDB).where(ModelDB.model_id == model_id)
        result = await session.exec(statement)

        if not result.first():
            raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")

        # Delete existing checkpoint and create new one
        delete_stmt = select(CheckpointDB).where(
            CheckpointDB.model_id == model_id,
            CheckpointDB.checkpoint_id == checkpoint_id,
            CheckpointDB.checkpoint_type == checkpoint_type,
        )
        existing = (await session.exec(delete_stmt)).first()
        if existing:
            await session.delete(existing)
            await session.flush()

        # Re-add the new checkpoint
        checkpoint_db = CheckpointDB(
            model_id=model_id,
            checkpoint_id=checkpoint_id,
            checkpoint_type=checkpoint_type,
            status=CheckpointStatus.PENDING,
        )
        session.add(checkpoint_db)
        await session.flush()


def evict_old_sampler_checkpoints_sync(
    session: Session,
    model_id: str,
    config: EngineConfig,
):
    """Delete oldest sampler checkpoints if count exceeds max_sampler_checkpoints_per_model.

    Called AFTER successfully saving a new sampler checkpoint.
    Deletes the database entry, the checkpoint archive, and the extracted lora directory (if exists).

    Args:
        session: SQLModel Session (sync)
        model_id: The model ID
        config: EngineConfig with checkpoints_base, max_sampler_checkpoints_per_model, external_inference_lora_base
    """
    max_count = config.max_sampler_checkpoints_per_model

    # Get all COMPLETED sampler checkpoints for this model, ordered by creation time (oldest first)
    statement = (
        select(CheckpointDB)
        .where(CheckpointDB.model_id == model_id)
        .where(CheckpointDB.checkpoint_type == types.CheckpointType.SAMPLER)
        .where(CheckpointDB.status == CheckpointStatus.COMPLETED)
        .order_by(CheckpointDB.created_at.asc())
    )
    result = session.exec(statement)
    checkpoints_list = result.all()

    # If we have more than max_count, delete the oldest ones
    if len(checkpoints_list) > max_count:
        to_delete = checkpoints_list[: len(checkpoints_list) - max_count]
        for checkpoint in to_delete:
            checkpoint_id = checkpoint.checkpoint_id

            # Delete checkpoint archive from disk
            checkpoint_path = (
                config.checkpoints_base / model_id / "sampler_weights" / f"{checkpoint_id}.tar.gz"
            )
            try:
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    log.info("deleted sampler checkpoint file", component="checkpoint", path=str(checkpoint_path))
            except Exception as e:
                log.warning("failed to delete checkpoint file", component="checkpoint", path=str(checkpoint_path), error=str(e))

            # Delete sidecar JSON
            sidecar_path = checkpoint_path.with_suffix("").with_suffix(".json")
            try:
                if sidecar_path.exists():
                    sidecar_path.unlink()
                    log.info("deleted sampler checkpoint sidecar", component="checkpoint", path=str(sidecar_path))
            except Exception as e:
                log.warning("failed to delete checkpoint sidecar", component="checkpoint", path=str(sidecar_path), error=str(e))

            # Delete extracted lora directory (used by external inference / vLLM)
            if config.external_inference_lora_base:
                lora_dir = config.external_inference_lora_base / f"{model_id}_{checkpoint_id}"
                try:
                    if lora_dir.exists():
                        shutil.rmtree(lora_dir)
                        log.info("deleted extracted lora directory", component="checkpoint", path=str(lora_dir))
                except Exception as e:
                    log.warning("failed to delete lora directory", component="checkpoint", path=str(lora_dir), error=str(e))

            # Delete from database
            session.delete(checkpoint)
            log.info("evicted sampler checkpoint", component="checkpoint", model_id=model_id, checkpoint_id=checkpoint_id)

        session.commit()


def evict_old_latest_training_checkpoints_sync(
    session: Session,
    model_id: str,
    checkpoint_id: str,
    config: EngineConfig,
):
    """Delete oldest 'latest_*' training checkpoints if count exceeds MAX_LATEST_TRAINING_CHECKPOINTS.

    Only evicts checkpoints whose checkpoint_id starts with LATEST_CHECKPOINT_PREFIX.
    Called AFTER successfully saving a new latest training checkpoint.

    Args:
        session: SQLModel Session (sync)
        model_id: The model ID
        checkpoint_id: The checkpoint ID being saved (to check if it's a latest_* checkpoint)
        config: EngineConfig with checkpoints_base
    """
    max_count = MAX_LATEST_TRAINING_CHECKPOINTS
    if not checkpoint_id.startswith(LATEST_CHECKPOINT_PREFIX):
        return

    statement = (
        select(CheckpointDB)
        .where(CheckpointDB.model_id == model_id)
        .where(CheckpointDB.checkpoint_type == types.CheckpointType.TRAINING)
        .where(CheckpointDB.status == CheckpointStatus.COMPLETED)
        .where(CheckpointDB.checkpoint_id.startswith(LATEST_CHECKPOINT_PREFIX))
        .order_by(CheckpointDB.created_at.asc())
    )
    result = session.exec(statement)
    checkpoints_list = result.all()

    if len(checkpoints_list) > max_count:
        to_delete = checkpoints_list[: len(checkpoints_list) - max_count]
        for checkpoint in to_delete:
            ckpt_id = checkpoint.checkpoint_id
            checkpoint_path = config.checkpoints_base / model_id / f"{ckpt_id}.tar.gz"
            sidecar_path = config.checkpoints_base / model_id / f"{ckpt_id}.json"
            for path in (checkpoint_path, sidecar_path):
                try:
                    if path.exists():
                        path.unlink()
                        log.info("deleted latest training checkpoint file", component="checkpoint", path=str(path))
                except Exception as e:
                    log.warning("failed to delete checkpoint file", component="checkpoint", path=str(path), error=str(e))

            session.delete(checkpoint)
            log.info("evicted latest training checkpoint", component="checkpoint", model_id=model_id, checkpoint_id=ckpt_id)

        session.commit()
