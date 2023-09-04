#include "wkt_parser.h"
#include <iostream>
#include <exception>

typedef boost::geometry::model::d2::point_xy<double> Boost_Point;

WKTParser::WKTParser(const cv::Size &size) : m_size(size) {

}

WKTParser::~WKTParser() {

}

bool WKTParser::ParsePoint(const std::string &src, cv::Point *point_ptr) {
    Boost_Point bp;
    try {
        boost::geometry::read_wkt(src, bp);
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        return false;
    }

    m_points.emplace_back(bp.x() * m_size.width, bp.y() * m_size.height);

    if (point_ptr) {
        point_ptr->x = bp.x() * m_size.width;
        point_ptr->y = bp.y() * m_size.height;
    }

    return true;
}

bool WKTParser::ParseLinestring(const std::string &src, VectorPoint *pvp) {
    boost::geometry::model::linestring<Boost_Point> linestring;
    try {
        boost::geometry::read_wkt(src, linestring);
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        return false;
    }

    VectorPoint vp;
    for (auto &item : linestring) {
        vp.emplace_back(item.x() * m_size.width, item.y() * m_size.height);
    }
    m_lines.emplace_back(vp);

    if (pvp) {
        *pvp = vp;
    }

    return true;
}

bool WKTParser::ParsePolygon(const std::string &src, VectorPoint *pvp) {
    boost::geometry::model::polygon<Boost_Point> ploygon;
    try {
        boost::geometry::read_wkt(src, ploygon);
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
        return false;
    }

    VectorPoint vp;
    for (auto &item : ploygon.outer()) {
        vp.emplace_back(item.x() * m_size.width, item.y() * m_size.height);
    }
    m_polygons.emplace_back(vp);

    if (pvp) {
        *pvp = vp;
    }

    return true;
}

bool WKTParser::InPolygons(const cv::Point &point) {
    if (empty()) return false;

    for (auto iter = m_polygons.cbegin(); iter != m_polygons.cend(); iter++) {
        if (WKTParser::InPolygon(*iter, point)) {
            return true;
        }
    }

    return false;
}

bool WKTParser::InPolygons(const cv::Rect &rect) {
    if (empty()) return false;

    for (auto iter = m_polygons.cbegin(); iter != m_polygons.cend(); iter++) {
        if (WKTParser::InPolygon(*iter, rect)) {
            return true;
        }
    }

    return false;
}

bool WKTParser::Polygon2Rect(const VectorPoint &polygon, cv::Rect &rect) {
    int min_x, min_y, max_x, max_y;
    min_x = min_y = std::numeric_limits<int>::max();
    max_x = max_y = std::numeric_limits<int>::min();

    for (size_t i = 0; i < polygon.size(); i++) {
        if (polygon[i].x < min_x) min_x = polygon[i].x;
        if (polygon[i].x > max_x) max_x = polygon[i].x;
        if (polygon[i].y < min_y) min_y = polygon[i].y;
        if (polygon[i].y > max_y) max_y = polygon[i].y;
    }

    rect.x = min_x;
    rect.y = min_y;
    rect.width = max_x - min_x;
    rect.height = max_y - min_y;

    return true;
}

bool WKTParser::InPolygon(const VectorPoint &polygon, const cv::Point &point) {
    cv::Point2f pf(point.x, point.y);
    return (cv::pointPolygonTest(polygon, pf, false) >= 0);
}

bool WKTParser::InPolygon(const VectorPoint &polygon, const cv::Rect &rect) {
    return (WKTParser::InPolygon(polygon, rect.tl()) &&
            WKTParser::InPolygon(polygon, cv::Point(rect.x + rect.width, rect.y)) &&
            WKTParser::InPolygon(polygon, rect.br()) &&
            WKTParser::InPolygon(polygon, cv::Point(rect.x, rect.y + rect.height)));
}
