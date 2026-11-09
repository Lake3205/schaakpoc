import pytest

from schaakgui.projects import DatasetEntry, Project, ProjectManager


def test_create_save_load_delete(tmp_path):
    manager = ProjectManager(tmp_path / ".projects")
    project = manager.create("chess one")
    assert manager.list_projects() == ["chess one"]
    project.train_folders = [DatasetEntry("/data/a", True), DatasetEntry("/data/b", False)]
    project.seed = 7
    project.augmentation.hflip = True
    project.quota.patience = 9
    project.training.epochs = 3
    manager.save(project)

    loaded = manager.load("chess one")
    assert loaded.checked_train_folders() == ["/data/a"]
    assert loaded.seed == 7 and loaded.augmentation.hflip
    assert loaded.quota.patience == 9 and loaded.training.epochs == 3

    (manager.project_dir("chess one") / "checkpoints").mkdir(parents=True)
    manager.delete("chess one")
    assert manager.list_projects() == []
    assert not manager.project_dir("chess one").exists()


def test_duplicate_and_invalid_names(tmp_path):
    manager = ProjectManager(tmp_path)
    manager.create("p1")
    with pytest.raises(ValueError):
        manager.create("p1")
    for bad in ("../evil", "", "a/b", ".hidden", "x.", "x.json"):
        with pytest.raises(ValueError):
            manager.create(bad)


def test_from_dict_ignores_unknown_keys():
    data = Project(name="x").to_dict()
    data["unknown"] = 1
    data["quota"]["unknown"] = 2
    assert Project.from_dict(data).name == "x"


def test_pre_v21_test_folders_become_validation_folders():
    data = Project(name="old").to_dict()
    del data["val_folders"]
    data["test_folders"] = [{"path": "/data/val", "checked": True}]
    project = Project.from_dict(data)
    assert project.checked_val_folders() == ["/data/val"] and project.test_folders == []
    roundtrip = Project.from_dict(project.to_dict())
    assert roundtrip.checked_val_folders() == ["/data/val"] and roundtrip.test_folders == []
