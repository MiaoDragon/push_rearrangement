#include "motoman_utils.h"


void generate_robot_geoms(const mjModel *m, std::vector<int> &robot_geoms)
{
    robot_geoms.clear();
    for (int i=0; i<m->ngeom; i++)
    {
        int rootid = m->body_rootid[m->geom_bodyid[i]];
        if (strcmp(mj_id2name(m, mjOBJ_BODY, rootid), "motoman_base") == 0)
        {
            robot_geoms.push_back(i);
        }
    }
}

void generate_exclude_pairs(const mjModel* m, IntPairSet& exclude_pairs)
{
    exclude_pairs.clear();
    /* exclude the pairs of the model */
    for (int i=0; i<m->nexclude; i++)
    {
        int a = m->exclude_signature[i] >> 16;
        int b = m->exclude_signature[i] & 0xFFFF;
        exclude_pairs.insert(std::make_pair(a,b));
    }

    /* exclude defined links */
    std::vector<const char*> exclude_links = {"arm_right_link_7_t", "base", "base_mount", "right_driver", "right_coupler", "right_spring_link", "right_follower", "right_pad",
                                              "left_driver", "left_coupler", "left_spring_link", "left_follower", "left_pad"};
    for (int i=0; i<exclude_links.size(); i++)
    {
        for (int j=i+1; j<exclude_links.size(); j++)
        {
            int a = mj_name2id(m, mjOBJ_BODY, exclude_links[i]);
            int b = mj_name2id(m, mjOBJ_BODY, exclude_links[j]);
            exclude_pairs.insert(std::make_pair(a,b));
        }
    }
}


void generate_collision_pairs(const mjModel* m,
                              const std::vector<int>& robot_geoms, const IntPairSet& exclude_pairs,
                              IntPairVector& collision_pairs)
{
    collision_pairs.clear();

    for (int i=0; i<robot_geoms.size(); i++)
    {
        for (int j=i+1; j<robot_geoms.size(); j++)
        {
            int geom1 = robot_geoms[i];
            int geom2 = robot_geoms[j];
            int body1 = m->geom_bodyid[geom1];
            int body2 = m->geom_bodyid[geom2];
            if ((m->body_parentid[body1] == body2) || (m->body_parentid[body2] == body1))
            {
                continue;
            }
            if (body1 == body2) continue;
            if (exclude_pairs.find(std::make_pair(body1,body2)) != exclude_pairs.end()) continue;
            if (exclude_pairs.find(std::make_pair(body2,body1)) != exclude_pairs.end()) continue;
            if ((m->geom_contype[geom1] & m->geom_conaffinity[geom2]) || (m->geom_contype[geom2] & m->geom_conaffinity[geom1]))
            {
                collision_pairs.push_back(std::make_pair(body1,body2));
            }
        }
    }
}
